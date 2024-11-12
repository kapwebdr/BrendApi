import psutil
import json
import asyncio
from fastapi import WebSocket
import GPUtil
import docker
import cachetools
import time
import httpx
from typing import Optional, Dict, Any, Callable, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime

@dataclass
class StreamResponse:
    type: str  # "log", "status", "error"
    content: str
    metadata: Optional[Dict[str, Any]] = None

class LogCallbacks:
    def __init__(self,
                 on_log: Optional[Callable[[str, Dict[str, Any]], None]] = None,
                 on_complete: Optional[Callable[[], None]] = None,
                 on_error: Optional[Callable[[Exception], None]] = None,
                 should_stop: Optional[Callable[[], bool]] = None):
        self.on_log = on_log
        self.on_complete = on_complete
        self.on_error = on_error
        self.should_stop = should_stop or (lambda: False)

class SystemMonitor:
    def __init__(self):
        self.client = docker.from_env()
        # Cache pour les conteneurs et stats
        self.last_valid_containers = []
        self.last_valid_stats = {}
        self.last_valid_metrics = {}
        # Tâche de mise à jour périodique
        self.update_task = None
        self._stop_event = asyncio.Event()
        # Démarrer la mise à jour périodique
        self.start_periodic_update()

    @staticmethod
    async def get_system_metrics():
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_freq = psutil.cpu_freq()
        print(cpu_percent)
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
        try:
            # Essayer d'abord le cache TTL
            if container_id in self.last_valid_stats:
                return self.last_valid_stats[container_id]
            
            container = self.client.containers.get(container_id)
            if container.status != "running":
                stats = {
                    "cpu_percent": 0,
                    "memory_percent": 0,
                    "memory_usage": 0,
                    "memory_limit": 0
                }
            else:
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
                
                stats = {
                    "cpu_percent": round(cpu_percent, 2),
                    "memory_percent": round(mem_percent, 2),
                    "memory_usage": mem_usage,
                    "memory_limit": mem_limit
                }
            
            # Mettre à jour les deux caches
            self.last_valid_stats[container_id] = stats
            return stats
            
        except Exception as e:
            print(f"Erreur stats container {container_id}: {str(e)}")
            # Retourner les dernières stats valides si disponibles
            return self.last_valid_stats.get(container_id, {
                "cpu_percent": 0,
                "memory_percent": 0,
                "memory_usage": 0,
                "memory_limit": 0
            })

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

    async def get_cached_containers_info(self):
        """Renvoie les données en cache des conteneurs"""
        return self.last_valid_containers

    async def get_containers_info(self) -> list:
        """Récupère la liste des conteneurs avec leurs statistiques"""
        try:
            containers = self.client.containers.list(all=True)
            container_info = []
            
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
                    
                    # Récupérer les stats si le conteneur est en cours d'exécution
                    if container.status == "running":
                        try:
                            # Utiliser la méthode existante get_container_stats
                            info["stats"] = await self.get_container_stats(container.id)
                            
                            # Vérifier si le service est prêt
                            try:
                                async with httpx.AsyncClient() as client:
                                    #response = await client.get(f"http://{container.name}:8000/ready", timeout=2.0)
                                    #info["ready"] = response.status_code == 200
                                    info["ready"] = True
                            except:
                                info["ready"] = False
                        except Exception as e:
                            print(f"Erreur stats pour {container.name}: {str(e)}")
                            info["stats"] = None
                            info["ready"] = False
                    else:
                        info["stats"] = None
                        info["ready"] = False
                    
                    container_info.append(info)
            
            # Mettre à jour le cache avec les nouvelles données
            self.last_valid_containers = container_info
            return container_info
            
        except Exception as e:
            print(f"Erreur get_containers_info: {str(e)}")
            # En cas d'erreur, retourner le cache
            return self.last_valid_containers

    async def get_container_logs(self, container_id: str, lines: int = 100) -> Dict:
        """Récupère les derniers logs d'un conteneur"""
        try:
            container = self.client.containers.get(container_id)
            logs = container.logs(tail=lines, timestamps=True)
            log_entries = []
            
            for log in logs.decode('utf-8').strip().split('\n'):
                if log:
                    try:
                        timestamp, message = log.split(' ', 1)
                    except ValueError:
                        timestamp = datetime.now().isoformat()
                        message = log

                    log_entries.append({
                        "timestamp": timestamp,
                        "message": message
                    })
            
            return {
                "status": "success",
                "container_id": container_id,
                "logs": log_entries
            }

        except Exception as e:
            return {
                "status": "error",
                "container_id": container_id,
                "error": str(e)
            }

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

    def stop(self):
        """Arrête le streaming en cours"""
        if not self._stop_event.is_set():
            self._stop_event.set()

    def reset(self):
        """Réinitialise le moniteur"""
        self._stop_event.clear()

    async def update_storage(self):
        """Met à jour les données dans le service storage"""
        try:
            async with httpx.AsyncClient() as client:
                # Mise à jour des conteneurs
                containers = await self.get_containers_info()
                await client.post("http://storage:8000/v1/storage/set", json={
                    "key": "containers_info",
                    "value": containers,
                    "collection": "monitor"
                })

                # Mise à jour des métriques système
                metrics = await self.get_system_metrics()
                print(metrics)
                await client.post("http://storage:8000/v1/storage/set", json={
                    "key": "system_metrics",
                    "value": metrics,
                    "collection": "monitor"
                })

                # Mettre à jour le cache local
                self.last_valid_containers = containers
                self.last_valid_metrics = metrics

        except Exception as e:
            print(f"Erreur mise à jour storage: {str(e)}")

    async def periodic_update(self):
        """Tâche périodique de mise à jour"""
        while not self._stop_event.is_set():
            await self.update_storage()
            await asyncio.sleep(60)  # Attendre 1 minute

    def start_periodic_update(self):
        """Démarre la tâche de mise à jour périodique"""
        if self.update_task is None:
            self.update_task = asyncio.create_task(self.periodic_update())

    def stop_periodic_update(self):
        """Arrête la tâche de mise à jour périodique"""
        if self.update_task:
            self._stop_event.set()
            self.update_task.cancel()
            self.update_task = None

    async def get_cached_system_metrics(self):
        """Renvoie toujours les données en cache"""
        return self.last_valid_metrics

    def __del__(self):
        """Nettoyage à la destruction de l'instance"""
        self.stop_periodic_update()