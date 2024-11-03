from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
import os
from typing import List, Optional, Any, Dict
import uuid
from Kapweb.session import SessionManager, UserSession
import redis
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

# Configuration Redis
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    decode_responses=True
)

session_manager = SessionManager()
current_file = os.path.realpath(__file__)

async def get_session(x_session_id: Optional[str] = Header(None)) -> UserSession:
    if not x_session_id:
        x_session_id = str(uuid.uuid4())
        session = session_manager.create_session(x_session_id)
    else:
        session = session_manager.get_session(x_session_id)
        if not session:
            session = session_manager.create_session(x_session_id)
    return session

def load_models(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data

def format_prompt(messages, system_message, prompt_template=None):
    history = []
    current_prompt = ""
    
    for message in messages:
        if message['role'] in ['user', 'human']:
            current_prompt = message['content']
        elif message['role'] in ['assistant', 'ai']:
            history.append(message['content'])
    
    final_prompt = f"{system_message}\n\n"
    return final_prompt

def loadLlm(model_name):
    model_path = os.path.join(os.path.dirname(current_file), "Cache", "LlamaCppModel", "bartowski",
                           "Llama-3.2-3B-Instruct-GGUF","Llama-3.2-3B-Instruct-Q4_0.gguf")
    if not os.path.exists(model_path):
        return None

    n_gpu_layers = 1
    n_batch = 4096
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=n_batch,
        f16_kv=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        streaming=True,
        verbose=True,
    )
    return llm

@app.post("/models")
async def list_models(session: UserSession = Depends(get_session)):
    try:
        available_models = load_models("/app/models.json")  # Chemin absolu dans le conteneur
        return JSONResponse(
            content={"models": list(available_models.keys())},
            headers={"X-Session-ID": session.session_id}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du chargement des modèles : {str(e)}"
        )

@app.post("/load_model")
async def load_model(request: Request, session: UserSession = Depends(get_session)):
    data = await request.json()
    available_models = load_models(os.path.join(os.path.dirname(current_file), "models.json"))
    
    if "model_name" not in data:
        raise HTTPException(status_code=400, detail="Nom du modèle requis")
    
    model_name = data["model_name"]
    if model_name not in available_models:
        raise HTTPException(status_code=404, detail="Modèle non trouvé")

    llm = loadLlm(available_models[model_name])
    if llm:
        session.llm = llm
        session.current_model = model_name
        session.loaded_model_config = available_models[model_name]
        return JSONResponse(
            content={"status": "loaded"},
            headers={"X-Session-ID": session.session_id}
        )
    else:
        raise HTTPException(status_code=500, detail="Échec du chargement du modèle")

@app.post("/generate")
async def generate(request: Request, session: UserSession = Depends(get_session)):
    data = await request.json()
    required_fields = {"model", "messages"}
    if not all(field in data for field in required_fields):
        raise HTTPException(status_code=400, detail="Configuration LLM invalide")
    
    formatted_prompt = format_prompt(
        messages=data["messages"],
        system_message=data.get("system", "Tu es Brenda, mon assistante, secrétaire personnelle.")
    )
    
    # Stockage de l'état de la session dans Redis
    session_key = f"session:{session.session_id}"
    redis_client.hset(session_key, mapping={
        "current_model": data["model"],
        "last_prompt": formatted_prompt
    })
    
    if data.get("stream", False):
        async def generate_stream():
            llm = loadLlm(data["model"])
            for chunk in llm.stream(formatted_prompt):
                if chunk:
                    yield f'data: {chunk}\n\n'
            yield 'data: [DONE]\n\n'

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={"X-Session-ID": session.session_id}
        )
    else:
        llm = loadLlm(data["model"])
        content = llm.invoke(formatted_prompt)
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

@app.get("/session")
async def session_status(session: UserSession = Depends(get_session)):
    return JSONResponse(
        content={
            "session_id": session.session_id,
            "current_model": session.current_model,
            "has_model_loaded": session.llm is not None
        },
        headers={"X-Session-ID": session.session_id}
    )

@app.post("/stop")
async def stop_generation(session: UserSession = Depends(get_session)):
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
