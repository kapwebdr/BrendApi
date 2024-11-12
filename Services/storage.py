from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from Kapweb.storage import StorageManager
from Kapweb.history import HistoryManager
from Kapweb.services import ServiceHelper
from Kapweb.session import SessionManager, UserSession
import os
import uuid
from typing import Optional
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

service = ServiceHelper("storage")
session_manager = SessionManager()
storage = StorageManager(
    backend=os.getenv('STORAGE_BACKEND', 'redis'),
    redis_host=os.getenv('REDIS_HOST', 'localhost'),
    redis_port=os.getenv('REDIS_PORT', 6379)
)
history_storage = StorageManager(
    backend='mongo',  # Force MongoDB pour l'historique
    mongo_url=os.getenv('MONGO_URL', 'mongodb://localhost:27017'),
    mongo_db=os.getenv('MONGO_DB', 'default')
)
history = HistoryManager(storage_manager=history_storage)

async def get_session(x_session_id: Optional[str] = Header(None)) -> UserSession:
    if not x_session_id:
        x_session_id = str(uuid.uuid4())
        session = session_manager.create_session(x_session_id)
    else:
        session = session_manager.get_session(x_session_id)
        if not session:
            session = session_manager.create_session(x_session_id)
    return session

@app.post("/v1/storage/set")
async def store_data(request: Request):
    data = await request.json()
    if not all(k in data for k in ["key", "value"]):
        raise HTTPException(status_code=400, detail="Key et value requis")
    
    result = await storage.store_data(
        key=data["key"],
        value=data["value"],
        collection=data.get("collection", "default")
    )
    
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.get("/v1/storage/get/{collection}/{key}")
async def get_data(collection: str, key: str):
    result = await storage.get_data(key, collection)
    if result is None:
        raise HTTPException(status_code=404, detail="Données non trouvées")
    if isinstance(result, dict) and result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.delete("/v1/storage/delete/{collection}/{key}")
async def delete_data(collection: str, key: str):
    result = await storage.delete_data(key, collection)
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.get("/v1/storage/list/{collection}")
async def list_data(collection: str, pattern: str = "*"):
    result = await storage.list_data(collection, pattern)
    if isinstance(result, dict) and result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.post("/v1/storage/search")
async def search_data(request: Request):
    data = await request.json()
    if "query" not in data:
        raise HTTPException(status_code=400, detail="Query requise")
    
    result = await storage.search_data(
        query=data["query"],
        collection=data.get("collection", "default"),
        n_results=data.get("n_results", 10)
    )
    
    if isinstance(result, dict) and result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.get("/v1/history/conversation/{id}")
async def get_conversation(id: str, session: UserSession = Depends(get_session)):
    """Récupère une conversation complète avec tous ses messages"""
    result = await history.get_conversation(id)
    
    # Vérification que la conversation appartient à la session
    if isinstance(result, dict):
        if result.get("status") == "error":
            raise HTTPException(status_code=404, detail=result["message"])
        if result.get("session_id") != session.session_id:
            raise HTTPException(status_code=403, detail="Non autorisé à accéder à cette conversation")
    
    return JSONResponse(
        content={
            "id": result["id"],
            "title": result["title"],
            "created_at": result["created_at"],
            "messages": result["messages"]
        },
        headers={"X-Session-ID": session.session_id}
    )

@app.get("/v1/history/conversations")
async def list_conversations(session: UserSession = Depends(get_session)):
    """Liste les conversations de la session, triées par date de création"""
    result = await history.list_conversations(session.session_id)
    if isinstance(result, dict) and result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    
    return JSONResponse(
        content={
            "conversations": [
                {
                    "id": conv["id"],
                    "title": conv["title"],
                    "created_at": conv["created_at"]
                }
                for conv in result
            ]
        },
        headers={"X-Session-ID": session.session_id}
    )

@app.post("/v1/history/conversation")
async def save_conversation(request: Request, session: UserSession = Depends(get_session)):
    """Sauvegarde un message dans une conversation"""
    data = await request.json()
    
    if "role" not in data or "message" not in data:
        raise HTTPException(status_code=400, detail="role et message requis")
    
    result = await history.save_conversation(
        session_id=session.session_id,
        id=data.get("id"),
        role=data["role"],
        message=data["message"],
        metadata=data.get("metadata")
    )
    
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    
    return JSONResponse(
        content={
            "status": "success",
            "id": result["id"],
            "title": result["title"],
            "message_id": result["message_id"]
        },
        headers={"X-Session-ID": session.session_id}
    )

@app.delete("/v1/history/conversation/{id}")
async def delete_conversation(id: str, session: UserSession = Depends(get_session)):
    """Supprime une conversation complète"""
    conv = await history.get_conversation(id)
    if isinstance(conv, dict):
        if conv.get("status") == "error":
            raise HTTPException(status_code=404, detail="Conversation non trouvée")
        if conv.get("session_id") != session.session_id:
            raise HTTPException(status_code=403, detail="Non autorisé à supprimer cette conversation")
    
    result = await history.delete_conversation(id)
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    
    return JSONResponse(
        content=result,
        headers={"X-Session-ID": session.session_id}
    )

@app.delete("/v1/history/session/{session_id}")
async def delete_session_history(session_id: str, session: UserSession = Depends(get_session)):
    """Supprime toutes les conversations d'une session"""
    if session_id != session.session_id:
        raise HTTPException(status_code=403, detail="Non autorisé à supprimer l'historique d'une autre session")
    
    result = await history.delete_by_session(session_id)
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    
    return JSONResponse(
        content=result,
        headers={"X-Session-ID": session.session_id}
    )

@app.get("/ready")
async def ready():
    """Endpoint indiquant que le service est prêt"""
    return await service.check_ready()

@app.post("/v1/storage/files/index")
async def index_file(request: Request):
    """Indexe un fichier ou dossier dans ChromaDB"""
    data = await request.json()
    if not all(k in data for k in ["path", "content"]):
        raise HTTPException(status_code=400, detail="Path et content requis")
    
    result = await storage.store_data(
        key=data["path"],
        value={
            "path": data["path"],
            "type": data.get("type", "file"),
            "content": data["content"],
            "size": data.get("size"),
            "metadata": data.get("metadata", {}),
            "created_at": datetime.now().isoformat()
        },
        collection="files_index"
    )
    
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.get("/v1/storage/files/index")
async def list_indexed_files(type: Optional[str] = None):
    """Liste les fichiers indexés"""
    result = await storage.list_data("files_index", type=type)
    if isinstance(result, dict) and result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return {"items": result}

@app.post("/v1/storage/files/search")
async def search_indexed_files(request: Request):
    """Recherche dans les fichiers indexés"""
    data = await request.json()
    if "query" not in data:
        raise HTTPException(status_code=400, detail="Query requise")
    
    where = {"type": data["type"]} if "type" in data else None
    result = await storage.search_data(
        query=data["query"],
        collection="files_index",
        n_results=data.get("n_results", 10),
        where=where
    )
    
    if isinstance(result, dict) and result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return {"results": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)