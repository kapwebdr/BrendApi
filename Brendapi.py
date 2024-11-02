from fastapi import FastAPI, Request, HTTPException, Depends, Header, BackgroundTasks, WebSocket
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import os
from typing import List, Optional, Any, Dict
import uuid
from starlette.requests import Request
from starlette.responses import Response
from Kapweb.llm import (
    load_models, 
    loadLlm, 
    format_prompt, 
    brenda_system,
    generate_stream
)
from Kapweb.session import SessionManager, UserSession
from Kapweb.huggingface import download_model
from Kapweb.media import media_generator, media_analyzer, load_media_models
import base64
from PIL import Image
import io
from Kapweb.trad import translator
from Kapweb.system_monitor import SystemMonitor
from enum import Enum
from Kapweb.speech import speech_processor

app = FastAPI()

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

current_file = os.path.realpath(__file__)
session_manager = SessionManager()

# Fonction pour obtenir ou créer une session
async def get_session(x_session_id: Optional[str] = Header(None)) -> UserSession:
    if not x_session_id:
        x_session_id = str(uuid.uuid4())
        session = session_manager.create_session(x_session_id)
    else:
        session = session_manager.get_session(x_session_id)
        if not session:
            session = session_manager.create_session(x_session_id)
    return session

class AITool(str, Enum):
    # LLM
    LLM = "llm"
    LIST_MODELS = "list_models"
    LOAD_MODEL = "load_model"
    SESSION_STATUS = "session_status"
    STOP_GENERATION = "stop_generation"
    
    # Image
    IMAGE_GENERATION = "image_generation"
    IMAGE_REFINEMENT = "image_refinement"
    IMAGE_ANALYSIS = "image_analysis"
    OCR = "ocr"
    LIST_IMAGE_MODELS = "list_image_models"
    
    # Traduction
    TRANSLATION = "translation"
    LIST_LANGUAGES = "list_languages"
    
    # HTTP et Media
    URL_ANALYZE = "url_analyze"
    URL_EXTRACT = "url_extract"
    URL_STREAM = "url_stream"
    YOUTUBE_STREAM = "youtube_stream"
    
    # Audio
    TEXT_TO_SPEECH = "text_to_speech"
    SPEECH_TO_TEXT = "speech_to_text"
    LIST_SPEECH_MODELS = "list_speech_models"
    
    # Système
    SYSTEM_METRICS = "system_metrics"

class AIRequest(BaseModel):
    tool: AITool
    config: Dict[str, Any]

@app.middleware("http")
async def catch_disconnections(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        if "Broken pipe" in str(e) or "Connection reset by peer" in str(e):
            session_id = request.headers.get("X-Session-ID")
            if session_id:
                session = session_manager.get_session(session_id)
                if session and session.llm:
                    try:
                        session.llm.stop()
                    except:
                        pass
        raise e

@app.post("/v1/ai/process")
async def process_ai_request(
    request: AIRequest,
    background_tasks: BackgroundTasks,
    session: UserSession = Depends(get_session)
):
    print(f"\n=== Nouvelle requête AI: {request.tool} ===")
    print(f"Configuration: {request.config}")

    try:
        if request.tool == AITool.LIST_MODELS:
            available_models = load_models(os.path.join(os.path.dirname(current_file), "models.json"))
            return JSONResponse(
                content={"models": list(available_models.keys())},
                headers={"X-Session-ID": session.session_id}
            )

        elif request.tool == AITool.LOAD_MODEL:
            available_models = load_models(os.path.join(os.path.dirname(current_file), "models.json"))
            if "model_name" not in request.config:
                raise HTTPException(status_code=400, detail="Nom du modèle requis")
            
            model_name = request.config["model_name"]
            if model_name not in available_models:
                raise HTTPException(status_code=404, detail="Modèle non trouvé")

            async def stream_response():
                async for chunk in download_model(available_models[model_name]):
                    yield chunk
                
                llm = loadLlm(available_models[model_name])
                if llm:
                    session.llm = llm
                    session.current_model = model_name
                    session.loaded_model_config = available_models[model_name]
                    yield f'data: {{"status": "loaded"}}\n\n'
                else:
                    yield f'data: {{"error": "Échec du chargement du modèle"}}\n\n'

            return StreamingResponse(
                stream_response(), 
                media_type="text/event-stream",
                headers={"X-Session-ID": session.session_id}
            )

        elif request.tool == AITool.SESSION_STATUS:
            return JSONResponse(
                content={
                    "session_id": session.session_id,
                    "current_model": session.current_model,
                    "has_model_loaded": session.llm is not None
                },
                headers={"X-Session-ID": session.session_id}
            )

        elif request.tool == AITool.STOP_GENERATION:
            if not session.llm:
                raise HTTPException(
                    status_code=400,
                    detail="Aucun modèle n'est chargé pour cette session"
                )
            
            try:
                session.cleanup()
                return JSONResponse(
                    content={"status": "stopped"},
                    headers={"X-Session-ID": session.session_id}
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Erreur lors de l'arrêt de la génération : {str(e)}"
                )

        elif request.tool == AITool.LIST_IMAGE_MODELS:
            available_models = media_generator.get_available_models()
            return JSONResponse(
                content={"models": available_models},
                headers={"X-Session-ID": session.session_id}
            )

        elif request.tool == AITool.LIST_LANGUAGES:
            return JSONResponse(
                content={"languages": translator.get_available_languages()},
                headers={"X-Session-ID": session.session_id}
            )

        elif request.tool == AITool.LLM:
            available_models = load_models(os.path.join(os.path.dirname(current_file), "models.json"))
            
            required_fields = {"model", "messages"}
            if not all(field in request.config for field in required_fields):
                raise HTTPException(status_code=400, detail="Configuration LLM invalide")
            
            formatted_prompt = format_prompt(
                messages=request.config["messages"],
                system_message=request.config.get("system", brenda_system)
            )
            
            if request.config.get("stream", False):
                return StreamingResponse(
                    generate_stream(formatted_prompt, session, request.config["model"], available_models),
                    media_type="text/event-stream",
                    headers={"X-Session-ID": session.session_id}
                )
            else:
                content = session.llm.invoke(formatted_prompt)
                return JSONResponse(
                    content={
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": content
                                }
                            }
                        ]
                    },
                    headers={"X-Session-ID": session.session_id}
                )

        elif request.tool == AITool.IMAGE_GENERATION:
            if "prompt" not in request.config:
                raise HTTPException(status_code=400, detail="Prompt requis pour la génération d'image")
            
            try:
                await media_generator.init_model(request.config.get("model_type", "sdxl/turbo"))
                
                return StreamingResponse(
                    media_generator.generate_image(
                        prompt=request.config["prompt"],
                        negative_prompt=request.config.get("negative_prompt", ""),
                        width=request.config.get("width", 1024),
                        height=request.config.get("height", 1024),
                        steps=request.config.get("steps", 20)
                    ),
                    media_type="text/event-stream",
                    headers={"X-Session-ID": session.session_id}
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Erreur lors de la génération d'image : {str(e)}"
                )

        elif request.tool == AITool.IMAGE_REFINEMENT:
            required_fields = {"image", "prompt"}
            if not all(field in request.config for field in required_fields):
                raise HTTPException(status_code=400, detail="Configuration de raffinement invalide")
            
            try:
                image_data = base64.b64decode(
                    request.config["image"].split(',')[1] 
                    if ',' in request.config["image"] 
                    else request.config["image"]
                )
                image = Image.open(io.BytesIO(image_data))
                
                await media_generator.init_model("sdxl/refiner")
                
                return StreamingResponse(
                    media_generator.refine_image_data(
                        image=image,
                        prompt=request.config["prompt"],
                        negative_prompt=request.config.get("negative_prompt", ""),
                        strength=request.config.get("strength", 0.3),
                        steps=request.config.get("steps", 20)
                    ),
                    media_type="text/event-stream",
                    headers={"X-Session-ID": session.session_id}
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Erreur lors du raffinement de l'image : {str(e)}"
                )

        elif request.tool == AITool.IMAGE_ANALYSIS:
            required_fields = {"image", "labels"}
            if not all(field in request.config for field in required_fields):
                raise HTTPException(status_code=400, detail="Configuration d'analyse invalide")
            
            try:
                image_data = base64.b64decode(
                    request.config["image"].split(',')[1] 
                    if ',' in request.config["image"] 
                    else request.config["image"]
                )
                image = Image.open(io.BytesIO(image_data))
                
                result = await media_analyzer.analyze_image_data(
                    image=image,
                    labels=request.config["labels"]
                )
                return JSONResponse(
                    content=result,
                    headers={"X-Session-ID": session.session_id}
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Erreur lors de l'analyse de l'image : {str(e)}"
                )

        elif request.tool == AITool.OCR:
            if "image" not in request.config:
                raise HTTPException(status_code=400, detail="Image requise pour l'OCR")
            
            try:
                image_data = base64.b64decode(
                    request.config["image"].split(',')[1] 
                    if ',' in request.config["image"] 
                    else request.config["image"]
                )
                image = Image.open(io.BytesIO(image_data))
                
                text = media_analyzer.ocr_image_data(image)
                return JSONResponse(
                    content={"text": text},
                    headers={"X-Session-ID": session.session_id}
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Erreur lors de l'extraction du texte : {str(e)}"
                )

        elif request.tool == AITool.TRANSLATION:
            required_fields = {"text", "from_lang", "to_lang"}
            if not all(field in request.config for field in required_fields):
                raise HTTPException(status_code=400, detail="Configuration de traduction invalide")
            
            if not translator.is_valid_language_pair(
                request.config["from_lang"], 
                request.config["to_lang"]
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Paire de langues non supportée"
                )
            
            try:
                result = await translator.translate(
                    text=request.config["text"],
                    from_lang=request.config["from_lang"],
                    to_lang=request.config["to_lang"]
                )
                
                if "error" in result:
                    raise HTTPException(
                        status_code=500,
                        detail=result["error"]
                    )
                    
                return JSONResponse(
                    content=result,
                    headers={"X-Session-ID": session.session_id}
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Erreur lors de la traduction : {str(e)}"
                )

        elif request.tool == AITool.TEXT_TO_SPEECH:
            required_fields = {"text"}
            if not all(field in request.config for field in required_fields):
                raise HTTPException(status_code=400, detail="Configuration TTS invalide")
            
            await speech_processor.init_tts()
            return StreamingResponse(
                speech_processor.text_to_speech(
                    text=request.config["text"],
                    voice_path=request.config.get("voice_path"),
                    language=request.config.get("language", "fr")
                ),
                media_type="text/event-stream",
                headers={"X-Session-ID": session.session_id}
            )

        elif request.tool == AITool.SPEECH_TO_TEXT:
            if "audio" not in request.config:
                raise HTTPException(status_code=400, detail="Audio requis pour la transcription")
            
            try:
                audio_data = base64.b64decode(
                    request.config["audio"].split(',')[1] 
                    if ',' in request.config["audio"] 
                    else request.config["audio"]
                )
                
                await speech_processor.init_stt()
                result = await speech_processor.speech_to_text(audio_data)
                
                return JSONResponse(
                    content=result,
                    headers={"X-Session-ID": session.session_id}
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Erreur lors de la transcription : {str(e)}"
                )

        elif request.tool == AITool.LIST_SPEECH_MODELS:
            return JSONResponse(
                content={"models": speech_processor.get_available_models()},
                headers={"X-Session-ID": session.session_id}
            )

        elif request.tool == AITool.SYSTEM_METRICS:
            metrics = await SystemMonitor.get_system_metrics()
            return JSONResponse(
                content=metrics,
                headers={"X-Session-ID": session.session_id}
            )

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Outil non reconnu : {request.tool}"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement de la requête {request.tool}: {str(e)}"
        )

if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run(
            "Brendapi:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            reload_dirs=[".", "Kapweb"],
            reload_delay=2
        )
    except ImportError:
        print("Erreur : uvicorn n'est pas installé.")
