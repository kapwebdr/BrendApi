from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from Kapweb.trad import translator
from Kapweb.services import ServiceHelper
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

service = ServiceHelper("translation")

@app.get("/v1/ai/translation/languages")
async def list_languages():
    return JSONResponse(content={"languages": translator.get_available_languages()})

@app.post("/v1/ai/translation/translate")
async def translate_text(request: Request):
    data = await request.json()
    required_fields = {"text", "from_lang", "to_lang"}
    if not all(field in data for field in required_fields):
        raise HTTPException(status_code=400, detail="Configuration de traduction invalide")
    
    if not translator.is_valid_language_pair(data["from_lang"], data["to_lang"]):
        raise HTTPException(status_code=400, detail="Paire de langues non supportée")
    
    try:
        request_id = str(uuid.uuid4())
        await service.store_data(
            key=f"translate_{request_id}",
            value={
                "text": data["text"],
                "from_lang": data["from_lang"],
                "to_lang": data["to_lang"],
                "timestamp": str(datetime.now())
            },
            collection="translation_requests"
        )
        
        result = await translator.translate(
            text=data["text"],
            from_lang=data["from_lang"],
            to_lang=data["to_lang"]
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        await service.store_data(
            key=f"translate_result_{request_id}",
            value=result,
            collection="translation_results"
        )
            
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ready")
async def ready():
    """Endpoint indiquant que le service est prêt"""
    return await service.check_ready({
        "translator": translator.is_ready
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)