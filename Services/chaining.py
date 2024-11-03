from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from Kapweb.services import ServiceHelper
import httpx
import base64
import json
import uuid
from datetime import datetime
import asyncio

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

service = ServiceHelper("chaining")

async def transcribe_audio(audio_data: str, model_size: str = "small"):
    """Transcrit l'audio en texte via le service speech"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://speech:8000/v1/ai/speech/speech-to-text",
            json={
                "audio": audio_data,
                "model_size": model_size
            }
        )
        return response.json()

async def generate_llm_response(text: str):
    """Génère une réponse LLM en streaming"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://llm:8000/v1/ai/generate",
            json={
                "model": "Llama-3.2-3B-Instruct",  # modèle par défaut
                "messages": [{"role": "user", "content": text}],
                "stream": True,
                "format_type": "speech"  # utilise le format optimisé pour la synthèse vocale
            },
            timeout=None
        )
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                yield line
            await asyncio.sleep(0)  

async def text_to_speech(text: str, voice: str = "elise", language: str = "fr"):
    """Convertit le texte en audio via le service speech"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://speech:8000/v1/ai/speech/text-to-speech",
            json={
                "text": text,
                "voice": voice,
                "language": language,
                "stream": False
            },
            timeout=None
        )
        audio_data = await response.aread()  
        # {"status": "completed", "audio": # Lit tout le contenu de la réponse
        return audio_data

async def conversation_stream(text_input: str = None, audio_input: str = None):
    """Gère le flux de conversation complet"""
    try:
        # Si audio fourni, transcription
        if audio_input:
            transcription = await transcribe_audio(audio_input)
            text_input = transcription["text"]

        # Stockage de l'entrée
        request_id = str(uuid.uuid4())
        await service.store_data(
            key=f"input_{request_id}",
            value={
                "text": text_input,
                "had_audio": bool(audio_input),
                "timestamp": str(datetime.now())
            },
            collection="conversation_history"
        )

        # Génération de la réponse LLM
        buffer = ""
        async for line in generate_llm_response(text_input):
            if line.startswith("data: "):
                line = line[6:]
                if line != '[DONE]':
                    data = json.loads(line)
                    if isinstance(data, dict) and "text" in data:
                        buffer += data["text"]
                        if isinstance(data, dict) and "pause" in data and data["pause"] in ["long", "short"]:
                            if buffer.strip():
                                yield f'data: {{"type": "text", "content": {json.dumps(buffer)}}}\n\n'
                                audio_data = await text_to_speech(buffer.strip())
                                audio_data = audio_data[6:]
                                yield f'data: {{"type": "audio", "content": "{audio_data}"}}\n\n'
                                buffer = ""

        # Traite le reste du buffer
        if buffer.strip():
            print('ICI :: ',buffer.strip())
            yield f'data: {{"type": "text", "content": {json.dumps(buffer)}}}\n\n'
            audio_data = await text_to_speech(buffer.strip())
            audio_data = audio_data[6:]
            yield f'data: {{"type": "audio", "content": "{audio_data}"}}\n\n'

        yield 'data: [DONE]\n\n'

    except Exception as e:
        yield f'data: {{"error": "{str(e)}"}}\n\n'
        yield 'data: [DONE]\n\n'

@app.post("/v1/ai/chaining/chat")
async def chat(request: Request):
    """Endpoint principal de conversation"""
    data = await request.json()
    
    if not (data.get("text") or data.get("audio")):
        raise HTTPException(
            status_code=400, 
            detail="Text ou audio requis"
        )

    return StreamingResponse(
        conversation_stream(
            text_input=data.get("text"),
            audio_input=data.get("audio")
        ),
        media_type="text/event-stream"
    )

@app.get("/ready")
async def ready():
    """Endpoint indiquant que le service est prêt"""
    try:
        # Vérifie que les services requis sont prêts
        async with httpx.AsyncClient() as client:
            llm_ready = await client.get("http://llm:8000/ready")
            speech_ready = await client.get("http://speech:8000/ready")
            
            if llm_ready.status_code == 200 and speech_ready.status_code == 200:
                return await service.check_ready()
            raise HTTPException(
                status_code=503, 
                detail="Services dépendants non prêts"
            )
    except Exception as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Service non prêt: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 