from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from Kapweb.speech import speech_processor
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

@app.post("/models")
async def list_models():
    return JSONResponse(content={"models": speech_processor.get_available_models()})

@app.post("/text-to-speech")
async def text_to_speech(request: Request):
    data = await request.json()
    if "text" not in data:
        raise HTTPException(status_code=400, detail="Texte requis pour la synthèse vocale")
    
    try:
        await speech_processor.init_tts()
        return StreamingResponse(
            speech_processor.text_to_speech(
                text=data["text"],
                voice_path=data.get("voice_path"),
                language=data.get("language", "fr")
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/speech-to-text")
async def speech_to_text(request: Request):
    data = await request.json()
    if "audio" not in data:
        raise HTTPException(status_code=400, detail="Audio requis pour la transcription")
    
    try:
        audio_data = base64.b64decode(
            data["audio"].split(',')[1] if ',' in data["audio"] else data["audio"]
        )
        
        await speech_processor.init_stt(data.get("model_size", "large-v3"))
        
        if data.get("stream", False):
            return StreamingResponse(
                speech_processor.speech_to_text_streaming(audio_data),
                media_type="text/event-stream"
            )
        else:
            result = await speech_processor.speech_to_text(audio_data)
            return JSONResponse(content=result)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 