from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from Kapweb.services import ServiceHelper
from Kapweb.llm import LLMCallbacks, llm_generator, format_chunk
from Kapweb.speech import SpeechCallbacks, speech_processor
import base64
import json
import uuid
from datetime import datetime
import asyncio
import io
import soundfile as sf

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

async def process_tts(text: str):
    """Fonction auxiliaire pour traiter la synthèse vocale"""
    async for tts_response in speech_processor.text_to_speech(
        text=text.strip(),
        model_type="xtts_v2",
        voice="elise",
        language="fr",
        callbacks=SpeechCallbacks(
            on_complete=lambda audio: print(f"Audio généré pour: {text}")
        )
    ):
        if tts_response.type == "audio":
            return tts_response.content
    return None

async def conversation_stream(text_input: str = None, audio_input: str = None):
    """Gère le flux de conversation complet"""
    try:
        # Si audio fourni, transcription
        if audio_input:
            audio_data = base64.b64decode(
                audio_input.split(',')[1] if ',' in audio_input else audio_input
            )
            
            text_buffer = ""
            async for response in speech_processor.speech_to_text(
                audio_data=audio_data,
                model_size="small",
                callbacks=SpeechCallbacks(
                    on_segment=lambda segment: print(f"Segment transcrit: {segment}"),
                    on_complete=lambda text: print(f"Transcription terminée: {text}")
                )
            ):
                if response.type == "text":
                    text_buffer = response.content
                    text_input = text_buffer

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

        # Buffer pour accumuler le texte pour la synthèse vocale
        text_for_tts = ""
        tts_task = None

        async for llm_response in llm_generator.generate_stream(
            prompt=text_input,
            session=None,
            model_name="Llama-3.2-3B-Instruct",
            format_type="speech",
            callbacks=LLMCallbacks(
                on_chunk=lambda chunk, meta: print(f"Chunk généré: {chunk}", meta),
                on_complete=lambda text: print(f"Génération terminée: {text}")
            )
        ):
            if llm_response.type == "text":
                text_for_tts += llm_response.content
                # Envoie immédiatement le texte
                yield f'data: {{"type": "text", "content": "{format_chunk(llm_response.content)}"}}\n\n'

                # Si on a une pause, lance la génération audio en parallèle
                if llm_response.metadata and "pause" in llm_response.metadata:
                    if llm_response.metadata["pause"] in ["long", "short"]:
                        # Si une tâche TTS est en cours, attend sa fin
                        if tts_task and not tts_task.done():
                            try:
                                audio_content = await tts_task
                                if audio_content:
                                    yield f'data: {{"type": "audio", "content": "{audio_content}"}}\n\n'
                            except Exception as e:
                                print(f"Erreur TTS: {e}")

                        # Lance une nouvelle tâche TTS
                        text_to_process = text_for_tts.strip()
                        if text_to_process:
                            tts_task = asyncio.create_task(process_tts(text_to_process))
                            text_for_tts = ""

        # Traite le dernier morceau de texte s'il en reste
        if text_for_tts.strip():
            # Attend la fin de la tâche précédente si elle existe
            if tts_task and not tts_task.done():
                try:
                    audio_content = await tts_task
                    if audio_content:
                        yield f'data: {{"type": "audio", "content": "{audio_content}"}}\n\n'
                except Exception as e:
                    print(f"Erreur TTS finale: {e}")

            # Traite le dernier morceau
            audio_content = await process_tts(text_for_tts.strip())
            if audio_content:
                yield f'data: {{"type": "audio", "content": "{audio_content}"}}\n\n'

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