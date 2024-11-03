from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from Kapweb.media import MediaGenerator,media_generator, media_analyzer
from Kapweb.services import ServiceHelper
import base64
import uuid
from datetime import datetime

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

service = ServiceHelper("media")
media_generator = MediaGenerator(models_config_path="/app/serve")
@app.post("/v1/ai/media/url/analyze")
async def analyze_url(request: Request):
    data = await request.json()
    if "url" not in data:
        raise HTTPException(status_code=400, detail="URL requise")
    
    try:
        request_id = str(uuid.uuid4())
        await service.store_data(
            key=f"analyze_{request_id}",
            value={
                "url": data["url"],
                "timestamp": str(datetime.now())
            },
            collection="media_requests"
        )
        
        result = await media_analyzer.analyze_url(data["url"])
        
        await service.store_data(
            key=f"analyze_result_{request_id}",
            value=result,
            collection="media_results"
        )
        
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ai/media/url/extract")
async def extract_url(request: Request):
    data = await request.json()
    if "url" not in data:
        raise HTTPException(status_code=400, detail="URL requise")
    
    try:
        request_id = str(uuid.uuid4())
        await service.store_data(
            key=f"extract_{request_id}",
            value={
                "url": data["url"],
                "timestamp": str(datetime.now())
            },
            collection="media_requests"
        )
        
        result = await media_analyzer.extract_url_content(data["url"])
        
        await service.store_data(
            key=f"extract_result_{request_id}",
            value=result,
            collection="media_results"
        )
        
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ai/media/url/stream")
async def stream_url(request: Request):
    data = await request.json()
    if "url" not in data:
        raise HTTPException(status_code=400, detail="URL requise")
    
    try:
        request_id = str(uuid.uuid4())
        await service.store_data(
            key=f"stream_{request_id}",
            value={
                "url": data["url"],
                "timestamp": str(datetime.now())
            },
            collection="media_requests"
        )
        
        return StreamingResponse(
            media_analyzer.stream_url(data["url"]),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ready")
async def ready():
    """Endpoint indiquant que le service est prêt"""
    return await service.check_ready()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)