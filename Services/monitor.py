from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from Kapweb.system_monitor import SystemMonitor
from Kapweb.services import ServiceHelper
import time

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = ServiceHelper("monitor")
system_monitor = SystemMonitor()

@app.get("/v1/monitor/containers")
async def list_containers():
    container_info = await system_monitor.get_containers_info()
    await service.store_data(
        key=f"containers_state_{int(time.time())}",
        value=container_info,
        collection="monitor_history"
    )
    return container_info

@app.get("/v1/monitor/system/stats")
async def get_system_stats():
    try:
        stats = await system_monitor.get_system_metrics()
        await service.store_data(
            key=f"system_stats_{int(time.time())}",
            value=stats,
            collection="system_stats"
        )
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/monitor/containers/{container_id}/logs")
async def get_container_logs(container_id: str, lines: int = 100):
    logs = await system_monitor.get_container_logs(container_id, lines)
    return {"logs": logs}

@app.post("/v1/monitor/containers/{container_id}/start")
async def start_container(container_id: str):
    success = await system_monitor.start_container(container_id)
    if success:
        return {"status": "started", "container_id": container_id}
    raise HTTPException(status_code=500, detail="Failed to start container")

@app.post("/v1/monitor/containers/{container_id}/stop")
async def stop_container(container_id: str):
    success = await system_monitor.stop_container(container_id)
    if success:
        return {"status": "stopped", "container_id": container_id}
    raise HTTPException(status_code=500, detail="Failed to stop container")

@app.post("/v1/monitor/containers/{container_id}/restart")
async def restart_container(container_id: str):
    success = await system_monitor.restart_container(container_id)
    if success:
        return {"status": "restarted", "container_id": container_id}
    raise HTTPException(status_code=500, detail="Failed to restart container")

@app.get("/ready")
async def ready():
    """Endpoint indiquant que le service est prêt"""
    return await service.check_ready()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)