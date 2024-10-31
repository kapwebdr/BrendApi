from fastapi import FastAPI, Request, HTTPException, Depends, Header, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import os
from typing import List, Optional
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
models = load_models(os.path.join(os.path.dirname(current_file), "models.json"))
session_manager = SessionManager()

# Charger les modèles d'images avec la bonne fonction
image_models = load_media_models(os.path.join(os.path.dirname(current_file), "image_models.json"))

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

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[dict]
    stream: bool = False
    system: str = brenda_system  # Valeur par défaut si non fournie

class ImageGenerationRequest(BaseModel):
    model_type: str = "sdxl/turbo"  # Par défaut utilise SDXL Turbo
    prompt: str
    negative_prompt: Optional[str] = ""
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    steps: Optional[int] = 20

class ImageAnalyzeRequest(BaseModel):
    image: str  # Image en base64
    labels: List[str]

class ImageOCRRequest(BaseModel):
    image: str  # Image en base64

class ImageRefineRequest(BaseModel):
    image: str  # Image source en base64
    prompt: str
    negative_prompt: Optional[str] = ""
    strength: Optional[float] = 0.3
    steps: Optional[int] = 20

class TranslationRequest(BaseModel):
    text: str
    from_lang: str
    to_lang: str

@app.middleware("http")
async def catch_disconnections(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        if "Broken pipe" in str(e) or "Connection reset by peer" in str(e):
            session_id = request.headers.get("X-Session-ID")
            if session_id:
                session = session_manager.get_session(session_id)
                if session and session.llm_instance:
                    try:
                        session.llm_instance.stop()
                    except:
                        pass
        raise e

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    session: UserSession = Depends(get_session)
):
    print("\n=== Nouvelle requête de chat ===")
    print(f"Modèle demandé: {request.model}")
    print(f"Session ID: {session.session_id}")
    print(f"Streaming: {request.stream}")
    print(f"Messages reçus: {request.messages}")
    print(f"Message système: {request.system}")

    # Vérifier si on doit charger un nouveau modèle
    if (not session.llm_instance or session.current_model != request.model) and request.model in models:
        print(f"\nChargement automatique du modèle {request.model}")
        async for chunk in download_model(models[request.model]):
            if isinstance(chunk, str) and "error" in chunk:
                raise HTTPException(status_code=500, detail="Erreur lors du téléchargement du modèle")
        
        chain, llm = loadLlm(models[request.model])
        if not chain or not llm:
            raise HTTPException(status_code=500, detail="Échec du chargement du modèle")
        
        session.llm_instance = chain
        session.llm = llm
        session.current_model = request.model
        session.loaded_model_config = models[request.model]
        print(f"Modèle {request.model} chargé avec succès")

    if not session.llm_instance:
        raise HTTPException(
            status_code=400,
            detail="Impossible de charger le modèle demandé"
        )

    print("\nFormatage du prompt...")
    formatted_prompt = format_prompt(
        messages=request.messages,
        system_message=request.system
    )
    
    print("\nPrompt formaté:")
    print(formatted_prompt)
    
    if request.stream:
        print("\nDémarrage du streaming...")
        config = {"configurable": {"session_id": session.session_id}}
        print(f"Configuration: {config}")
        
        response = StreamingResponse(
            generate_stream(formatted_prompt, session, request.model, models),
            media_type="text/event-stream",
            headers={"X-Session-ID": session.session_id}
        )
        
        background_tasks.add_task(session.cleanup)
        return response
    else:
        print("\nGénération en mode non-streaming...")
        try:
            config = {"configurable": {"session_id": session.session_id}}
            content = session.llm_instance.invoke(formatted_prompt, config=config)
            print(f"\nRéponse générée: {content}")
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
        except Exception as e:
            print(f"\nERREUR lors de la génération: {str(e)}")
            try:
                session.llm_instance.stop()
            except:
                pass
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models(session: UserSession = Depends(get_session)):
    return JSONResponse(
        content={"models": list(models.keys())},
        headers={"X-Session-ID": session.session_id}
    )

@app.post("/v1/models/{model_name}/load")
async def load_model(
    model_name: str,
    session: UserSession = Depends(get_session)
):
    if model_name not in models:
        raise HTTPException(status_code=404, detail="Modèle non trouvé")

    async def stream_response():
        async for chunk in download_model(models[model_name]):
            yield chunk
        
        chain, llm = loadLlm(models[model_name])  # Récupérer le tuple (chain, llm)
        if chain and llm:  # Vérifier les deux valeurs
            session.llm_instance = chain
            session.llm = llm
            session.current_model = model_name
            session.loaded_model_config = models[model_name]
            yield f"data: {{'status': 'loaded'}}\n\n"
        else:
            yield f"data: {{'error': 'Échec du chargement du modèle'}}\n\n"

    response = StreamingResponse(
        stream_response(), 
        media_type="text/event-stream",
        headers={"X-Session-ID": session.session_id}
    )
    return response

@app.get("/v1/session")
async def get_session_status(session: UserSession = Depends(get_session)):
    return JSONResponse(
        content={
            "session_id": session.session_id,
            "current_model": session.current_model,
            "has_model_loaded": session.llm_instance is not None
        },
        headers={"X-Session-ID": session.session_id}
    )

@app.post("/v1/stop")
async def stop_generation(session: UserSession = Depends(get_session)):
    if not session.llm_instance:
        raise HTTPException(
            status_code=400,
            detail="Aucun modèle n'est chargé pour cette session"
        )
    
    try:
        print(f"\nArrêt de la génération pour la session {session.session_id}")
        session.cleanup()  # Utilise la méthode cleanup existante
        return JSONResponse(
            content={"status": "stopped"},
            headers={"X-Session-ID": session.session_id}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'arrêt de la génération : {str(e)}"
        )

@app.post("/v1/images/generate")
async def generate_images(
    request: ImageGenerationRequest,
    background_tasks: BackgroundTasks,
    session: UserSession = Depends(get_session)
):
    print(f"\n=== Nouvelle requête de génération d'image ===")
    print(f"Type de modèle: {request.model_type}")
    print(f"Prompt: {request.prompt}")
    
    try:
        # Initialisation du modèle si nécessaire
        await media_generator.init_model(request.model_type)
        
        response = StreamingResponse(
            media_generator.generate_image(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                steps=request.steps
            ),
            media_type="text/event-stream",
            headers={"X-Session-ID": session.session_id}
        )
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération d'image : {str(e)}"
        )

@app.get("/v1/images/models")
async def list_image_models(session: UserSession = Depends(get_session)):
    """Liste les modèles de génération d'images disponibles"""
    models = media_generator.get_available_models()
    return JSONResponse(
        content={"models": models},
        headers={"X-Session-ID": session.session_id}
    )

@app.post("/v1/images/refine")
async def refine_image(
    request: ImageRefineRequest,
    background_tasks: BackgroundTasks,
    session: UserSession = Depends(get_session)
):
    """Raffine une image existante avec SDXL Refiner"""
    print(f"\n=== Nouvelle requête de raffinement d'image ===")
    print(f"Prompt: {request.prompt}")
    
    try:
        # Conversion du base64 en image
        image_data = base64.b64decode(request.image.split(',')[1] if ',' in request.image else request.image)
        image = Image.open(io.BytesIO(image_data))
        
        # Initialisation du modèle refiner
        await media_generator.init_model("sdxl/refiner")
        
        response = StreamingResponse(
            media_generator.refine_image_data(
                image=image,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                strength=request.strength,
                steps=request.steps
            ),
            media_type="text/event-stream",
            headers={"X-Session-ID": session.session_id}
        )
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du raffinement de l'image : {str(e)}"
        )

@app.post("/v1/images/analyze")
async def analyze_image(
    request: ImageAnalyzeRequest,
    session: UserSession = Depends(get_session)
):
    """Analyse une image avec CLIP"""
    print(f"\n=== Nouvelle requête d'analyse d'image ===")
    print(f"Labels: {request.labels}")
    
    try:
        # Conversion du base64 en image
        image_data = base64.b64decode(request.image.split(',')[1] if ',' in request.image else request.image)
        image = Image.open(io.BytesIO(image_data))
        
        result = await media_analyzer.analyze_image_data(
            image=image,
            labels=request.labels
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

@app.post("/v1/images/ocr")
async def ocr_image(
    request: ImageOCRRequest,
    session: UserSession = Depends(get_session)
):
    """Extrait le texte d'une image"""
    print(f"\n=== Nouvelle requête d'OCR ===")
    
    try:
        # Conversion du base64 en image
        image_data = base64.b64decode(request.image.split(',')[1] if ',' in request.image else request.image)
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

@app.get("/v1/translation/languages")
async def get_translation_languages(session: UserSession = Depends(get_session)):
    """Liste les langues disponibles pour la traduction"""
    return JSONResponse(
        content={"languages": translator.get_available_languages()},
        headers={"X-Session-ID": session.session_id}
    )

@app.post("/v1/translation/translate")
async def translate_text(
    request: TranslationRequest,
    session: UserSession = Depends(get_session)
):
    """Traduit un texte d'une langue à une autre"""
    print(f"\n=== Nouvelle requête de traduction ===")
    print(f"De: {request.from_lang}")
    print(f"Vers: {request.to_lang}")
    print(f"Texte: {request.text}")
    
    # Vérifier si la paire de langues est valide
    if not translator.is_valid_language_pair(request.from_lang, request.to_lang):
        raise HTTPException(
            status_code=400,
            detail="Paire de langues non supportée"
        )
    
    try:
        result = await translator.translate(
            text=request.text,
            from_lang=request.from_lang,
            to_lang=request.to_lang
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
