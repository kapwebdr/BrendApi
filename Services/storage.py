from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Any, Dict, List
import redis
import json
import os
import chromadb
from chromadb.config import Settings
from motor.motor_asyncio import AsyncIOMotorClient
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

# Configuration des backends de stockage
STORAGE_BACKEND = os.getenv('STORAGE_BACKEND', 'redis')  # 'redis', 'mongo', 'chroma'

# Redis configuration
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    decode_responses=True
)

# MongoDB configuration
mongo_client = AsyncIOMotorClient(os.getenv('MONGO_URL', 'mongodb://localhost:27017'))
mongo_db = mongo_client[os.getenv('MONGO_DB', 'brenda')]

# ChromaDB configuration
chroma_client = chromadb.PersistentClient(
    path=os.getenv('CHROMA_PERSIST_DIR', './chroma_storage'),
    settings=Settings(
        allow_reset=True,
        anonymized_telemetry=False
    )
)

class StorageService:
    def __init__(self, backend=STORAGE_BACKEND):
        self.backend = backend
        
    async def set(self, key: str, value: Any, collection: str = "default", metadata: Dict = None) -> bool:
        try:
            if self.backend == "redis":
                return await self._redis_set(key, value, collection)
            elif self.backend == "mongo":
                return await self._mongo_set(key, value, collection, metadata)
            elif self.backend == "chroma":
                return await self._chroma_set(key, value, collection, metadata)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erreur de stockage: {str(e)}")

    async def get(self, key: str, collection: str = "default") -> Any:
        try:
            if self.backend == "redis":
                return await self._redis_get(key, collection)
            elif self.backend == "mongo":
                return await self._mongo_get(key, collection)
            elif self.backend == "chroma":
                return await self._chroma_get(key, collection)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Clé non trouvée: {str(e)}")

    async def delete(self, key: str, collection: str = "default") -> bool:
        try:
            if self.backend == "redis":
                return await self._redis_delete(key, collection)
            elif self.backend == "mongo":
                return await self._mongo_delete(key, collection)
            elif self.backend == "chroma":
                return await self._chroma_delete(key, collection)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erreur de suppression: {str(e)}")

    async def list(self, collection: str = "default", pattern: str = "*") -> List[str]:
        try:
            if self.backend == "redis":
                return await self._redis_list(collection, pattern)
            elif self.backend == "mongo":
                return await self._mongo_list(collection, pattern)
            elif self.backend == "chroma":
                return await self._chroma_list(collection, pattern)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erreur de listage: {str(e)}")

    # Implémentations Redis
    async def _redis_set(self, key: str, value: Any, collection: str) -> bool:
        full_key = f"{collection}:{key}"
        return redis_client.set(full_key, json.dumps(value))

    async def _redis_get(self, key: str, collection: str) -> Any:
        full_key = f"{collection}:{key}"
        value = redis_client.get(full_key)
        if value is None:
            raise KeyError(f"Clé {key} non trouvée dans {collection}")
        return json.loads(value)

    async def _redis_delete(self, key: str, collection: str) -> bool:
        full_key = f"{collection}:{key}"
        return redis_client.delete(full_key) > 0

    async def _redis_list(self, collection: str, pattern: str) -> List[str]:
        full_pattern = f"{collection}:{pattern}"
        keys = redis_client.keys(full_pattern)
        return [k.split(':', 1)[1] for k in keys]

    # Implémentations MongoDB
    async def _mongo_set(self, key: str, value: Any, collection: str, metadata: Dict = None) -> bool:
        doc = {
            "_id": key,
            "value": value,
            "metadata": metadata or {},
            "updated_at": datetime.utcnow()
        }
        result = await mongo_db[collection].replace_one(
            {"_id": key}, doc, upsert=True
        )
        return result.acknowledged

    async def _mongo_get(self, key: str, collection: str) -> Any:
        doc = await mongo_db[collection].find_one({"_id": key})
        if doc is None:
            raise KeyError(f"Clé {key} non trouvée dans {collection}")
        return doc["value"]

    async def _mongo_delete(self, key: str, collection: str) -> bool:
        result = await mongo_db[collection].delete_one({"_id": key})
        return result.deleted_count > 0

    async def _mongo_list(self, collection: str, pattern: str) -> List[str]:
        cursor = mongo_db[collection].find(
            {"_id": {"$regex": pattern.replace("*", ".*")}}
        )
        return [doc["_id"] async for doc in cursor]

    # Implémentations ChromaDB
    async def _chroma_set(self, key: str, value: Any, collection: str, metadata: Dict = None) -> bool:
        try:
            chroma_collection = chroma_client.get_or_create_collection(name=collection)
            # Convertit la valeur en chaîne pour le stockage
            value_str = json.dumps(value)
            # Ajoute ou met à jour le document
            chroma_collection.upsert(
                ids=[key],
                documents=[value_str],
                metadatas=[metadata or {}]
            )
            return True
        except Exception as e:
            print(f"Erreur ChromaDB set: {str(e)}")
            return False

    async def _chroma_get(self, key: str, collection: str) -> Any:
        try:
            chroma_collection = chroma_client.get_or_create_collection(name=collection)
            result = chroma_collection.get(
                ids=[key],
                include=['documents']
            )
            if not result['documents']:
                raise KeyError(f"Clé {key} non trouvée dans {collection}")
            # Parse la chaîne JSON stockée
            return json.loads(result['documents'][0])
        except Exception as e:
            print(f"Erreur ChromaDB get: {str(e)}")
            raise

    async def _chroma_delete(self, key: str, collection: str) -> bool:
        try:
            chroma_collection = chroma_client.get_or_create_collection(name=collection)
            chroma_collection.delete(ids=[key])
            return True
        except Exception as e:
            print(f"Erreur ChromaDB delete: {str(e)}")
            return False

    async def _chroma_list(self, collection: str, pattern: str) -> List[str]:
        try:
            chroma_collection = chroma_client.get_or_create_collection(name=collection)
            # Récupère tous les IDs de la collection
            result = chroma_collection.get(include=['ids'])
            return result['ids']
        except Exception as e:
            print(f"Erreur ChromaDB list: {str(e)}")
            return []

storage_service = StorageService()

@app.post("/v1/storage/set")
async def set_value(request: Request):
    data = await request.json()
    if not all(k in data for k in ["key", "value"]):
        raise HTTPException(status_code=400, detail="key et value requis")
    
    collection = data.get("collection", "default")
    metadata = data.get("metadata", None)
    
    success = await storage_service.set(
        data["key"], 
        data["value"], 
        collection,
        metadata
    )
    return {"success": success}

@app.get("/v1/storage/get/{collection}/{key}")
async def get_value(collection: str, key: str):
    try:
        value = await storage_service.get(key, collection)
        return {"value": value}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.delete("/v1/storage/delete/{collection}/{key}")
async def delete_value(collection: str, key: str):
    success = await storage_service.delete(key, collection)
    return {"success": success}

@app.get("/v1/storage/list/{collection}")
async def list_keys(collection: str, pattern: str = "*"):
    keys = await storage_service.list(collection, pattern)
    return {"keys": keys}

@app.get("/ready")
async def ready():
    """Endpoint indiquant que le service est prêt"""
    try:
        if STORAGE_BACKEND == "redis":
            redis_client.ping()
        elif STORAGE_BACKEND == "mongo":
            await mongo_client.admin.command('ping')
        elif STORAGE_BACKEND == "chroma":
            chroma_client.heartbeat()
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 