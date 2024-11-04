from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from Kapweb.session import SessionManager, UserSession
from Kapweb.media import media_generator, media_analyzer
from typing import Optional
import base64
from PIL import Image
import io
import uuid
import pytesseract
from datetime import datetime
from Kapweb.services import ServiceHelper
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

service = ServiceHelper("image")
session_manager = SessionManager()
media_generator.models_config_path= '/app/Config/'

async def get_session(x_session_id: Optional[str] = Header(None)) -> UserSession:
    if not x_session_id:
        x_session_id = str(uuid.uuid4())
        session = session_manager.create_session(x_session_id)
    else:
        session = session_manager.get_session(x_session_id)
        if not session:
            session = session_manager.create_session(x_session_id)
    return session

@app.post("/v1/ai/image/models")
async def list_models():
    try:
        
        return JSONResponse(content={"models": media_generator.get_available_models()})
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du chargement des modèles : {str(e)}"
        )

@app.post("/v1/ai/image/ocr")
async def perform_ocr(request: Request):
    data = await request.json()
    if "image" not in data:
        raise HTTPException(status_code=400, detail="Image requise pour l'OCR")
    
    try:
        request_id = str(uuid.uuid4())
        
        # Décodage de l'image base64
        image_data = base64.b64decode(
            data["image"].split(',')[1] if ',' in data["image"] else data["image"]
        )
        image = Image.open(io.BytesIO(image_data))
        
        # Configuration de la langue pour l'OCR
        lang = data.get("lang", "fra")  # Par défaut en français
        
        # Stockage de la requête
        await service.store_data(
            key=f"ocr_{request_id}",
            value={
                "lang": lang,
                "timestamp": str(datetime.now())
            },
            collection="image_requests"
        )
        
        # Extraction du texte
        text = pytesseract.image_to_string(image, lang=lang)
        
        # Configuration avancée si spécifiée
        if data.get("advanced", False):
            result = {
                "text": text,
                "data": pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT),
                "boxes": pytesseract.image_to_boxes(image, lang=lang),
                "confidence": float(pytesseract.image_to_osd(image)['confidence']) if 'confidence' in pytesseract.image_to_osd(image) else None
            }
        else:
            result = {"text": text}
        
        # Stockage du résultat
        await service.store_data(
            key=f"ocr_result_{request_id}",
            value=result,
            collection="image_results"
        )
            
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'OCR : {str(e)}"
        )

@app.post("/v1/ai/image/generate")
async def generate_image(request: Request, session: UserSession = Depends(get_session)):
    data = await request.json()
    if "prompt" not in data:
        raise HTTPException(status_code=400, detail="Prompt requis pour la génération d'image")
    
    try:
        request_id = str(uuid.uuid4())
        await service.store_data(
            key=f"gen_{request_id}",
            value={
                "prompt": data["prompt"],
                "model_type": data.get("model_type", "sdxl/turbo"),
                "timestamp": str(datetime.now())
            },
            collection="image_requests"
        )
        
        await media_generator.init_model(data.get("model_type", "sdxl/turbo"))
        return StreamingResponse(
            media_generator.generate_image(
                prompt=data["prompt"],
                negative_prompt=data.get("negative_prompt", ""),
                width=data.get("width", 1024),
                height=data.get("height", 1024),
                steps=data.get("steps", 20)
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ai/image/refine")
async def refine_image(request: Request, session: UserSession = Depends(get_session)):
    data = await request.json()
    if not all(k in data for k in ["image", "prompt"]):
        raise HTTPException(status_code=400, detail="Image et prompt requis")
    
    try:
        request_id = str(uuid.uuid4())
        image_data = base64.b64decode(
            data["image"].split(',')[1] if ',' in data["image"] else data["image"]
        )
        image = Image.open(io.BytesIO(image_data))
        
        await service.store_data(
            key=f"refine_{request_id}",
            value={
                "prompt": data["prompt"],
                "timestamp": str(datetime.now())
            },
            collection="image_requests"
        )
        
        await media_generator.init_model("sdxl/refiner")
        return StreamingResponse(
            media_generator.refine_image_data(
                image=image,
                prompt=data["prompt"],
                negative_prompt=data.get("negative_prompt", ""),
                strength=data.get("strength", 0.3),
                steps=data.get("steps", 20)
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ai/image/analyze")
async def analyze_image(request: Request, session: UserSession = Depends(get_session)):
    data = await request.json()
    if not all(k in data for k in ["image", "labels"]):
        raise HTTPException(status_code=400, detail="Image et labels requis")
    
    try:
        request_id = str(uuid.uuid4())
        image_data = base64.b64decode(
            data["image"].split(',')[1] if ',' in data["image"] else data["image"]
        )
        image = Image.open(io.BytesIO(image_data))
        
        await service.store_data(
            key=f"analyze_{request_id}",
            value={
                "labels": data["labels"],
                "timestamp": str(datetime.now())
            },
            collection="image_requests"
        )
        
        result = await media_analyzer.analyze_image_data(
            image=image,
            labels=data["labels"]
        )
        
        await service.store_data(
            key=f"analyze_result_{request_id}",
            value=result,
            collection="image_results"
        )
        
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ready")
async def ready():
    """Endpoint indiquant que le service est prêt"""
    return await service.check_ready({
        "tesseract": pytesseract.get_tesseract_version
    })

@app.post("/v1/ai/image/load_model")
async def load_model_endpoint(request: Request, session: UserSession = Depends(get_session)):
    data = await request.json()
    
    if "model_type" not in data:
        raise HTTPException(status_code=400, detail="Type de modèle requis")
    
    model_type = data["model_type"]
    available_models = media_generator.get_available_models()
    
    if model_type not in available_models:
        raise HTTPException(status_code=404, detail="Modèle non trouvé")

    async def stream_response():
        try:
            model_config = media_generator.get_model_config(model_type)
            async for chunk in media_generator.load_model(model_type):
                yield chunk
            yield f'data: {{"status": "loaded"}}\n\n'
            
            # Stockage de l'état de la session
            await service.store_data(
                key=session.session_id,
                value={
                    "current_model": model_type,
                    "model_config": model_config
                },
                collection="image_sessions"
            )
            
        except Exception as e:
            yield f'data: {{"error": "Échec du chargement du modèle: {str(e)}"}}\n\n'
    
    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={"X-Session-ID": session.session_id}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)