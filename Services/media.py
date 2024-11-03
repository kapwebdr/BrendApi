from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from Kapweb.media import media_generator, media_analyzer
import base64
import uuid
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.post("/url/analyze")
async def analyze_url(request: Request):
    data = await request.json()
    if "url" not in data:
        raise HTTPException(status_code=400, detail="URL requise")
    
    try:
        result = await media_analyzer.analyze_url(data["url"])
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/url/extract")
async def extract_url(request: Request):
    data = await request.json()
    if "url" not in data:
        raise HTTPException(status_code=400, detail="URL requise")
    
    try:
        result = await media_analyzer.extract_url_content(data["url"])
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/url/stream")
async def stream_url(request: Request):
    data = await request.json()
    if "url" not in data:
        raise HTTPException(status_code=400, detail="URL requise")
    
    try:
        return StreamingResponse(
            media_analyzer.stream_url(data["url"]),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 