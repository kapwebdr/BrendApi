from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from Kapweb.trad import translator
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

@app.get("/languages")
async def list_languages():
    return JSONResponse(content={"languages": translator.get_available_languages()})

@app.post("/translate")
async def translate_text(request: Request):
    data = await request.json()
    required_fields = {"text", "from_lang", "to_lang"}
    if not all(field in data for field in required_fields):
        raise HTTPException(status_code=400, detail="Configuration de traduction invalide")
    
    if not translator.is_valid_language_pair(data["from_lang"], data["to_lang"]):
        raise HTTPException(status_code=400, detail="Paire de langues non supportée")
    
    try:
        result = await translator.translate(
            text=data["text"],
            from_lang=data["from_lang"],
            to_lang=data["to_lang"]
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 