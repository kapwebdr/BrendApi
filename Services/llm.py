from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from Kapweb.session import SessionManager, UserSession
from Kapweb.llm import get_prompt_template, load_models, loadLlm, format_prompt, brenda_system, llm_generator
from Kapweb.services import ServiceHelper
from Kapweb.huggingface import download_model
from typing import Optional
import os
import httpx
import json
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

service = ServiceHelper("llm")
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

@app.post("/v1/ai/models")
async def list_models(session: UserSession = Depends(get_session)):
    try:
        available_models = load_models("/app/Config/models.json")
        return JSONResponse(
            content={"models": list(available_models.keys())},
            headers={"X-Session-ID": session.session_id}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du chargement des modèles : {str(e)}"
        )

@app.post("/v1/ai/load_model")
async def load_model_endpoint(request: Request, session: UserSession = Depends(get_session)):
    data = await request.json()
    available_models = load_models("/app/Config/models.json")
    
    if "model_name" not in data:
        raise HTTPException(status_code=400, detail="Nom du modèle requis")
    
    model_name = data["model_name"]
    
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
        # Stockage de l'état de la session
        await service.store_data(
            key=session.session_id,
            value={
                "current_model": model_name,
                "loaded_model_config": available_models[model_name]
            },
            collection="llm_sessions"
        )
        
    return StreamingResponse(
            stream_response(), 
            media_type="text/event-stream",
            headers={"X-Session-ID": session.session_id}
        )
    
@app.post("/v1/ai/generate")
async def generate(request: Request, session: UserSession = Depends(get_session)):
    data = await request.json()
    required_fields = {"model"}
    if not all(field in data for field in required_fields):
        raise HTTPException(status_code=400, detail="Configuration LLM invalide")
    
    try:
        # Récupération de l'historique des messages via le service storage
        conversation_id = data.get("conversationId")
        if conversation_id:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://storage:8000/v1/history/conversation/{conversation_id}",
                    headers={"X-Session-ID": session.session_id}
                )
                
                if response.status_code == 200:
                    conversation = response.json()
                    messages = [
                        {
                            "role": msg["role"],
                            "content": msg["message"]
                        }
                        for msg in conversation.get("messages", [])
                        if not (
                            isinstance(msg.get("metadata"), dict) and 
                            msg["metadata"].get("isMedia", False)
                        )
                    ]
                    print(f"Messages récupérés: {len(messages)}")
                else:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail="Erreur lors de la récupération de la conversation"
                    )
        else:
            # Si pas de conversation_id, utilise les messages fournis directement
            messages = [{"role": "user", "content": data.get("prompt")}]
            if not messages:
                raise HTTPException(status_code=400, detail="Messages ou conversation_id requis")

        available_models = load_models("/app/Config/models.json")
        model_config = available_models[data["model"]]
        prompt_template = get_prompt_template(model_config.get("template"),'/app/Config/models_template.json')
        
        formatted_prompt = format_prompt(
            messages=messages,
            system_message=data.get("system", brenda_system),
            prompt_template=prompt_template
        )
        print("formatted_prompt :: ", formatted_prompt)

        # Stockage du prompt et de l'historique
        await service.store_data(
            key=f"{session.session_id}_last_prompt",
            value={
                "prompt": formatted_prompt,
                "messages": messages,
                "model": data["model"],
                "conversation_id": conversation_id
            },
            collection="llm_history"
        )
        
        if data.get("stream", False):
            async def stream_response():
                try:
                    async for response in llm_generator.generate_stream(
                        prompt=formatted_prompt,
                        session=session,
                        model_name=data["model"],
                        models=available_models,
                        format_type=data.get("format_type", "chunk")
                    ):
                        if response.type == "text":
                            if response.metadata:
                                yield f'data: {{"text": "{response.content}", "pause": "{response.metadata["pause"]}"}}\n\n'
                            else:
                                yield f'data: {{"text": "{response.content}"}}\n\n'
                        elif response.type == "status":
                            yield f'data: {{"status": "{response.content}", "text": "{response.metadata["complete_text"]}"}}\n\n'
                        elif response.type == "error":
                            yield f'data: {{"error": "{response.content}"}}\n\n'
                            
                except Exception as e:
                    yield f'data: {{"error": "{str(e)}"}}\n\n'

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
                headers={"X-Session-ID": session.session_id}
            )
        else:
            try:
                content = session.llm.invoke(formatted_prompt)
                
                # Stockage de la réponse
                await service.store_data(
                    key=f"{session.session_id}_last_response",
                    value=content,
                    collection="llm_responses"
                )
                
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
                raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/ai/session")
async def session_status(session: UserSession = Depends(get_session)):
    try:
        # Récupération de l'état de la session depuis le storage
        session_data = await service.get_data(session.session_id, "llm_sessions")
        return JSONResponse(
            content={
                "session_id": session.session_id,
                "current_model": session_data.get("current_model"),
                "has_model_loaded": session.llm is not None
            },
            headers={"X-Session-ID": session.session_id}
        )
    except:
        # Fallback si pas de données stockées
        return JSONResponse(
            content={
                "session_id": session.session_id,
                "current_model": session.current_model,
                "has_model_loaded": session.llm is not None
            },
            headers={"X-Session-ID": session.session_id}
        )

@app.post("/v1/ai/stop")
async def stop_generation(session: UserSession = Depends(get_session)):
    """Arrête la génération en cours"""
    try:
        llm_generator.stop()
        return JSONResponse(
            content={"status": "stopped"},
            headers={"X-Session-ID": session.session_id}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'arrêt de la génération : {str(e)}"
        )

@app.get("/ready")
async def ready():
    """Endpoint indiquant que le service est prêt"""
    return await service.check_ready()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
