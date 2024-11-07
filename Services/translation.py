from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse,JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from Kapweb.trad import StreamResponse, TranslationCallbacks, translator
from Kapweb.services import ServiceHelper
import uuid
from datetime import datetime
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

        async def stream_response():
            async for response in translator.translate(
                text=data["text"],
                from_lang=data["from_lang"],
                to_lang=data["to_lang"]
            ):
                if response.type == "status":
                    if response.content == "completed":
                        yield f'data: {{"status": "completed", "text": "{response.metadata["translated_text"]}", "from": "{response.metadata["from"]}", "to": "{response.metadata["to"]}", "model": "{response.metadata["model"]}"}}\n\n'
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

@app.get("/ready")
async def ready():
    """Endpoint indiquant que le service est prêt"""
    return await service.check_ready({
        "translator": translator.is_ready
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)