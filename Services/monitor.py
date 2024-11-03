from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import docker
import psutil
import os
from typing import List, Dict, Optional
from datetime import datetime
import httpx
import asyncio
import cachetools
import time
from Kapweb.services import ServiceHelper

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = ServiceHelper("monitor")
client = docker.from_env()

# Cache pour les conteneurs (30 secondes)
containers_cache = cachetools.TTLCache(maxsize=100, ttl=30)
# Cache pour les stats (5 secondes)
stats_cache = cachetools.TTLCache(maxsize=100, ttl=5)

def is_brenda_container(container_name: str) -> bool:
    return container_name.startswith('brenda_')

def get_cached_stats(container_id: str) -> Dict:
    """Récupère les stats du cache ou les calcule si nécessaire"""
    now = time.time()
    if container_id in stats_cache:
        return stats_cache[container_id]
    
    try:
        container = client.containers.get(container_id)
        if container.status != "running":
            return {
                "cpu_percent": 0,
                "memory_percent": 0,
                "memory_usage": 0,
                "memory_limit": 0
            }

        stats = container.stats(stream=False)
        cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - \
                   stats["precpu_stats"]["cpu_usage"]["total_usage"]
        system_delta = stats["cpu_stats"]["system_cpu_usage"] - \
                      stats["precpu_stats"]["system_cpu_usage"]
        cpu_percent = 0.0
        if system_delta > 0:
            cpu_percent = (cpu_delta / system_delta) * 100.0 * len(stats["cpu_stats"]["cpu_usage"]["percpu_usage"])
        
        mem_usage = stats["memory_stats"]["usage"]
        mem_limit = stats["memory_stats"]["limit"]
        mem_percent = (mem_usage / mem_limit) * 100.0
        
        result = {
            "cpu_percent": round(cpu_percent, 2),
            "memory_percent": round(mem_percent, 2),
            "memory_usage": mem_usage,
            "memory_limit": mem_limit
        }
        stats_cache[container_id] = result
        return result
    except:
        return {
            "cpu_percent": 0,
            "memory_percent": 0,
            "memory_usage": 0,
            "memory_limit": 0
        }

async def check_service_ready(container) -> bool:
    """Vérifie si un service est prêt en appelant son endpoint /ready"""
    try:
        # Utilise le nom du conteneur comme hostname dans le réseau Docker
        service_name = container.name.replace('brenda_', '')  # Retire le préfixe brenda_
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://{service_name}:8000/ready", timeout=2.0)
            print(f"http://{service_name}:8000/ready",container.name,response.status_code)
            return response.status_code == 200
    except Exception as e:
        print(f"Erreur check ready pour {container.name}: {str(e)}")
        return False

async def get_containers_info():
    """Récupère les informations des conteneurs avec mise en cache"""
    cache_key = 'containers_info'
    if cache_key in containers_cache:
        return containers_cache[cache_key]

    try:
        containers = client.containers.list(all=True)
        container_info = []
        ready_checks = []
        
        for container in containers:
            if is_brenda_container(container.name):
                # Création d'une tâche pour le check ready
                if container.status == "running":
                    ready_checks.append(check_service_ready(container))
                else:
                    ready_checks.append(None)
                
                container_info.append({
                    "id": container.id,
                    "name": container.name,
                    "status": container.status,
                    "image": container.image.tags[0] if container.image.tags else "none",
                    "created": container.attrs['Created'],
                    "ports": container.ports,
                })

        # Exécution parallèle des checks ready
        ready_results = await asyncio.gather(*[check for check in ready_checks if check is not None])
        ready_index = 0
        
        # Mise à jour des informations avec les résultats ready et stats
        for i, container in enumerate(container_info):
            if container["status"] == "running":
                container["ready"] = ready_results[ready_index]
                container["stats"] = get_cached_stats(container["id"])
                ready_index += 1
            else:
                container["ready"] = False
                container["stats"] = None

        containers_cache[cache_key] = container_info
        return container_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/monitor/containers")
async def list_containers():
    container_info = await get_containers_info()
    # Stocke les informations dans le service storage
    await service.store_data(
        key=f"containers_state_{int(time.time())}",
        value=container_info,
        collection="monitor_history"
    )
    return container_info

@app.get("/v1/monitor/system/stats")
async def get_system_stats():
    try:
        stats = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent,
            },
            "disk": {
                "total": psutil.disk_usage('/').total,
                "used": psutil.disk_usage('/').used,
                "free": psutil.disk_usage('/').free,
                "percent": psutil.disk_usage('/').percent,
            }
        }
        
        # Stocke les statistiques
        await service.store_data(
            key=f"system_stats_{int(time.time())}",
            value=stats,
            collection="system_stats"
        )
        
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/monitor/containers/{container_id}/start")
async def start_container(container_id: str):
    try:
        container = client.containers.get(container_id)
        container.start()
        return {"status": "started", "container_id": container_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/monitor/containers/{container_id}/stop")
async def stop_container(container_id: str):
    try:
        container = client.containers.get(container_id)
        container.stop()
        return {"status": "stopped", "container_id": container_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/monitor/containers/{container_id}/restart")
async def restart_container(container_id: str):
    try:
        container = client.containers.get(container_id)
        container.restart()
        return {"status": "restarted", "container_id": container_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/monitor/containers/{container_id}/logs")
async def get_container_logs(container_id: str, lines: int = 100):
    try:
        container = client.containers.get(container_id)
        logs = container.logs(tail=lines, timestamps=True).decode('utf-8')
        return {"logs": logs.split('\n')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/monitor/containers/{container_id}/stats")
async def get_container_statistics(container_id: str):
    try:
        container = client.containers.get(container_id)
        if container.status != "running":
            return {"error": "Container is not running"}
        return get_cached_stats(container_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/ready")
async def ready():
    """Endpoint indiquant que le service est prêt"""
    return await service.check_ready({
        "docker": client.ping,
        "psutil": lambda: bool(psutil.cpu_percent())
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 