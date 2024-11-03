from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from Kapweb.session import SessionManager, UserSession
from Kapweb.media import media_generator, media_analyzer
from typing import Optional
import base64
from PIL import Image
import io
import os
import json
import uuid
import pytesseract

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

session_manager = SessionManager()

async def get_session(x_session_id: Optional[str] = Header(None)) -> UserSession:
    if not x_session_id:
        x_session_id = str(uuid.uuid4())
        session = session_manager.create_session(x_session_id)
    else:
        session = session_manager.get_session(x_session_id)
        if not session:
            session = session_manager.create_session(x_session_id)
    return session

@app.post("/models")
async def list_models():
    try:
        with open("/app/image_models.json", 'r') as file:
            models = json.load(file)
        return JSONResponse(content={"models": models})
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du chargement des modèles : {str(e)}"
        )

@app.post("/ocr")
async def perform_ocr(request: Request):
    data = await request.json()
    if "image" not in data:
        raise HTTPException(status_code=400, detail="Image requise pour l'OCR")
    
    try:
        # Décodage de l'image base64
        image_data = base64.b64decode(
            data["image"].split(',')[1] if ',' in data["image"] else data["image"]
        )
        image = Image.open(io.BytesIO(image_data))
        
        # Configuration de la langue pour l'OCR
        lang = data.get("lang", "fra")  # Par défaut en français
        
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
            
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'OCR : {str(e)}"
        )

@app.post("/generate")
async def generate_image(request: Request, session: UserSession = Depends(get_session)):
    data = await request.json()
    if "prompt" not in data:
        raise HTTPException(status_code=400, detail="Prompt requis pour la génération d'image")
    
    try:
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

@app.post("/refine")
async def refine_image(request: Request, session: UserSession = Depends(get_session)):
    data = await request.json()
    if not all(k in data for k in ["image", "prompt"]):
        raise HTTPException(status_code=400, detail="Image et prompt requis")
    
    try:
        image_data = base64.b64decode(
            data["image"].split(',')[1] if ',' in data["image"] else data["image"]
        )
        image = Image.open(io.BytesIO(image_data))
        
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

@app.post("/analyze")
async def analyze_image(request: Request, session: UserSession = Depends(get_session)):
    data = await request.json()
    if not all(k in data for k in ["image", "labels"]):
        raise HTTPException(status_code=400, detail="Image et labels requis")
    
    try:
        image_data = base64.b64decode(
            data["image"].split(',')[1] if ',' in data["image"] else data["image"]
        )
        image = Image.open(io.BytesIO(image_data))
        
        result = await media_analyzer.analyze_image_data(
            image=image,
            labels=data["labels"]
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 