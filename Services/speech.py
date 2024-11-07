from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from Kapweb.speech import speech_processor, StreamResponse, SpeechCallbacks, SpeechProcessor
import base64
import uuid
import httpx
from datetime import datetime
from Kapweb.services import ServiceHelper
import soundfile
import torch
import json

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

service = ServiceHelper("speech")
speech_processor = SpeechProcessor(cache_dir="/app/Cache")

@app.post("/v1/ai/speech/models")
async def list_models():
    return JSONResponse(content={"models": speech_processor.get_available_models()})

@app.post("/v1/ai/speech/text-to-speech")
async def text_to_speech(request: Request):
    data = await request.json()
    if "text" not in data:
        raise HTTPException(status_code=400, detail="Texte requis pour la synthèse vocale")
    
    try:
        request_id = str(uuid.uuid4())
        
        # Validation des paramètres selon le type de modèle
        model_type = data.get("model_type", "xtts_v2")
        model_config = speech_processor.tts_config.get(model_type)
        
        if not model_config:
            raise HTTPException(status_code=400, detail=f"Type de modèle '{model_type}' non disponible")
        
        if model_config["requires_language"] and "language" not in data:
            raise HTTPException(status_code=400, detail=f"Le modèle {model_type} nécessite une langue")
            
        if model_config["type"] == "multi_speaker" and "voice" not in data:
            raise HTTPException(status_code=400, detail=f"Le modèle {model_type} nécessite une voix")

        # Stockage de la requête
        await service.store_data(
            key=f"tts_{request_id}",
            value={
                "text": data["text"],
                "model_type": model_type,
                "voice": data.get("voice"),
                "language": data.get("language"),
                "timestamp": str(datetime.now())
            },
            collection="speech_requests"
        )

        async def stream_response():
            async for response in speech_processor.text_to_speech(
                text=data["text"],
                model_type=model_type,
                voice=data.get("voice"),
                language=data.get("language"),
                stream=data.get("stream", True)
            ):
                if response.type == "audio":
                    # Encode l'audio en base64
                    audio_base64 = base64.b64encode(response.content).decode()
                    yield f'data: {{"audio": "{audio_base64}"}}\n\n'
                elif response.type == "status":
                    if response.content == "completed":
                        # Encode l'audio en base64 pour le completed aussi
                        audio_base64 = base64.b64encode(response.metadata["audio"]).decode()
                        yield f'data: {{"status": "completed", "audio": "{audio_base64}"}}\n\n'
                    else:
                        yield f'data: {{"status": "{response.content}"}}\n\n'
                elif response.type == "progress":
                    yield f'data: {{"progress": {response.content}, "message": "{response.metadata["message"]}"}}\n\n'
                elif response.type == "error":
                    yield f'data: {{"error": "{response.content}"}}\n\n'

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ai/speech/speech-to-text")
async def speech_to_text(request: Request):
    data = await request.json()
    if "audio" not in data:
        raise HTTPException(status_code=400, detail="Audio requis")
    
    try:
        request_id = str(uuid.uuid4())
        audio_data = base64.b64decode(
            data["audio"].split(',')[1] if ',' in data["audio"] else data["audio"]
        )

        # Validation du modèle STT
        model_size = data.get("model_size", "small")
        if model_size not in speech_processor.stt_config["whisper"]["models"]:
            raise HTTPException(status_code=400, detail=f"Taille de modèle '{model_size}' non disponible")

        # Stockage de la requête
        await service.store_data(
            key=f"stt_{request_id}",
            value={
                "model_size": model_size,
                "timestamp": str(datetime.now())
            },
            collection="speech_requests"
        )

        async def stream_response():
            async for response in speech_processor.speech_to_text(
                audio_data=audio_data,
                model_size=model_size
            ):
                if response.type == "text":
                    yield f'data: {{"text": "{response.content}"}}\n\n'
                elif response.type == "segment":
                    yield f'data: {{"type": "segment", "text": "{response.content}", "start": {response.metadata["start"]}, "end": {response.metadata["end"]}}}\n\n'
                elif response.type == "status":
                    if response.content == "completed":
                        yield f'data: {{"status": "completed", "text": "{response.metadata["text"]}"}}\n\n'
                    else:
                        yield f'data: {{"status": "{response.content}"}}\n\n'
                elif response.type == "progress":
                    yield f'data: {{"progress": {response.content}, "message": "{response.metadata["message"]}"}}\n\n'
                elif response.type == "error":
                    yield f'data: {{"error": "{response.content}"}}\n\n'

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ai/speech/stop")
async def stop_processing():
    """Arrête le traitement en cours"""
    try:
        speech_processor.stop()
        return JSONResponse(content={"status": "stopped"})
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'arrêt du traitement : {str(e)}"
        )

@app.get("/ready")
async def ready():
    """Endpoint indiquant que le service est prêt"""
    return await service.check_ready({
        "soundfile": lambda: bool(soundfile),
        "torch": lambda: bool(torch.cuda.is_available())
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)