import psutil
import json
import asyncio
from fastapi import WebSocket
import GPUtil
import docker
import cachetools
import time
import httpx

class SystemMonitor:
    def __init__(self):
        self.client = docker.from_env()
        # Cache pour les conteneurs (30 secondes)
        self.containers_cache = cachetools.TTLCache(maxsize=100, ttl=30)
        # Cache pour les stats (5 secondes)
        self.stats_cache = cachetools.TTLCache(maxsize=100, ttl=5)

    @staticmethod
    async def get_system_metrics():
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_freq = psutil.cpu_freq()
        
        # Mémoire
        memory = psutil.virtual_memory()
        
        # Disque
        disk = psutil.disk_usage('/')
        # GPU si disponible
        try:
            gpus = GPUtil.getGPUs()
            gpu_info = [{
                'id': gpu.id,
                'name': gpu.name,
                'load': gpu.load * 100,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'temperature': gpu.temperature
            } for gpu in gpus]
        except:
            gpu_info = []

        return {
            'cpu': {
                'percent': cpu_percent,
                'frequency_current': cpu_freq.current if cpu_freq else None,
                'frequency_max': cpu_freq.max if cpu_freq else None,
                'cores': psutil.cpu_count()
            },
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used
            },
            'disk': {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': disk.percent
            },
            'gpu': gpu_info
        }

    def is_brenda_container(self, container_name: str) -> bool:
        return container_name.startswith('brenda_')

    async def get_container_stats(self, container_id: str) -> dict:
        """Récupère les stats du cache ou les calcule si nécessaire"""
        if container_id in self.stats_cache:
            return self.stats_cache[container_id]
        
        try:
            container = self.client.containers.get(container_id)
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
                cpu_percent = (cpu_delta / system_delta) * 100.0 * stats["cpu_stats"]["online_cpus"]
            
            mem_usage = stats["memory_stats"]["usage"]
            mem_limit = stats["memory_stats"]["limit"]
            mem_percent = (mem_usage / mem_limit) * 100.0
            
            result = {
                "cpu_percent": round(cpu_percent, 2),
                "memory_percent": round(mem_percent, 2),
                "memory_usage": mem_usage,
                "memory_limit": mem_limit
            }
            self.stats_cache[container_id] = result
            return result
        except Exception as e:
            print(f"Erreur stats container {container_id}: {str(e)}")
            return {
                "cpu_percent": 0,
                "memory_percent": 0,
                "memory_usage": 0,
                "memory_limit": 0
            }

    async def check_service_ready(self, container_name: str) -> bool:
        """Vérifie si un service est prêt en appelant son endpoint /ready"""
        try:
            service_name = container_name.replace('brenda_', '')
            #async with httpx.AsyncClient() as client:
                # response = await client.get(f"http://{service_name}:8000/ready", timeout=1.0)
            return True
        except Exception as e:
            print(f"Erreur check ready pour {container_name}: {str(e)}")
            return False

    async def get_containers_info(self):
        """Récupère les informations des conteneurs avec mise en cache"""
        cache_key = 'containers_info'
        if cache_key in self.containers_cache:
            return self.containers_cache[cache_key]

        try:
            containers = self.client.containers.list(all=True)
            container_info = []
            ready_tasks = []
            
            for container in containers:
                if self.is_brenda_container(container.name):
                    info = {
                        "id": container.id,
                        "name": container.name,
                        "status": container.status,
                        "image": container.image.tags[0] if container.image.tags else "none",
                        "created": container.attrs['Created'],
                        "ports": container.ports,
                    }
                    
                    if container.status == "running":
                        info["stats"] = await self.get_container_stats(container.id)
                        ready_tasks.append(self.check_service_ready(container.name))
                    else:
                        info["stats"] = None
                        info["ready"] = False
                    
                    container_info.append(info)

            # Exécuter toutes les vérifications ready en parallèle
            if ready_tasks:
                ready_results = await asyncio.gather(*ready_tasks)
                ready_index = 0
                for container in container_info:
                    if container["status"] == "running":
                        container["ready"] = ready_results[ready_index]
                        ready_index += 1

            self.containers_cache[cache_key] = container_info
            return container_info
        except Exception as e:
            print(f"Erreur get_containers_info: {str(e)}")
            return []

    async def get_container_logs(self, container_id: str, lines: int = 100) -> list:
        """Récupère les logs d'un conteneur"""
        try:
            container = self.client.containers.get(container_id)
            logs = container.logs(tail=lines, timestamps=True).decode('utf-8')
            return logs.split('\n')
        except Exception as e:
            print(f"Erreur get_container_logs: {str(e)}")
            return []

    async def start_container(self, container_id: str) -> bool:
        try:
            container = self.client.containers.get(container_id)
            container.start()
            return True
        except Exception as e:
            print(f"Erreur start_container: {str(e)}")
            return False

    async def stop_container(self, container_id: str) -> bool:
        try:
            container = self.client.containers.get(container_id)
            container.stop()
            return True
        except Exception as e:
            print(f"Erreur stop_container: {str(e)}")
            return False

    async def restart_container(self, container_id: str) -> bool:
        try:
            container = self.client.containers.get(container_id)
            container.restart()
            return True
        except Exception as e:
            print(f"Erreur restart_container: {str(e)}")
            return False