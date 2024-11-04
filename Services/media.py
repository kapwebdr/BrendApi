from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from Kapweb.media import media_generator, media_analyzer
from Kapweb.http import http_processor
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
media_generator.models_config_path= '/app/Config/'

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
        
        async with http_processor as http:
            result = await http.analyze_url(data["url"])
        
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
                "selectors": data["selectors"],
                "timestamp": str(datetime.now())
            },
            collection="media_requests"
        )
        
        async with http_processor as http:
            return StreamingResponse(
                http.extract_content(data["url"], data["selectors"]),
                media_type="text/event-stream"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
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
        
        async with http_processor as http:
            return StreamingResponse(
                http.stream_content(data["url"]),
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