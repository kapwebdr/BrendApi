import psutil
import json
import asyncio
from fastapi import WebSocket
import GPUtil

class SystemMonitor:
    @staticmethod
    async def get_system_metrics():
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_freq = psutil.cpu_freq()
        
        # Mémoire
        memory = psutil.virtual_memory()
        
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
                'total': memory.total / (1024 ** 3),  # En GB
                'available': memory.available / (1024 ** 3),
                'percent': memory.percent,
                'used': memory.used / (1024 ** 3)
            },
            'gpu': gpu_info
        }