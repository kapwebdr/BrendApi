from typing import Optional, Any, Dict, List
import redis
import json
from pymongo import MongoClient
import chromadb
import os
import logging
import base64
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StorageManager:
    """
    Initialise le gestionnaire de stockage
    
    Args:
        backend: Type de backend ("redis", "mongo", "chroma")
        **kwargs: Configuration spécifique au backend
    """
    def __init__(self, backend: str = "redis", **kwargs):
        self.default_backend = backend
        self.config = kwargs
        logger.info(f"Initializing StorageManager with default backend: {backend}")
        logger.info(f"Configuration: {kwargs}")
        self.clients = {}
        self._init_backend(self.default_backend)

    def _init_backend(self, backend: str):
        """Initialise un backend spécifique si pas déjà fait"""
        if backend in self.clients:
            return self.clients[backend]

        try:
            if backend == "redis":
                redis_host = self.config.get('redis_host', 'redis')
                redis_port = int(self.config.get('redis_port', 6379))
                logger.info(f"Connecting to Redis at {redis_host}:{redis_port}")
                client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    decode_responses=True
                )
                client.ping()
                logger.info("Successfully connected to Redis")
                self.clients[backend] = client
                
            elif backend == "mongo":
                mongo_url = self.config.get('mongo_url', 'mongodb://mongo:27017')
                mongo_db = self.config.get('mongo_db', 'default')
                logger.info(f"Connecting to MongoDB at {mongo_url}")
                client = MongoClient(mongo_url)
                db = client[mongo_db]
                client.admin.command('ping')
                logger.info("Successfully connected to MongoDB")
                self.clients[backend] = {"client": client, "db": db}
                
            elif backend == "chroma":
                persist_dir = self.config.get('chroma_persist_dir', './chroma_db')
                logger.info(f"Initializing ChromaDB with persist_dir: {persist_dir}")
                client = chromadb.PersistentClient(path=persist_dir)
                logger.info("Successfully initialized ChromaDB")
                self.clients[backend] = client
                
        except Exception as e:
            logger.error(f"Failed to initialize {backend} backend: {str(e)}")
            raise

    def _get_client(self, backend: str = None):
        """Récupère ou initialise le client pour un backend donné"""
        current_backend = backend if backend in ["redis", "mongo", "chroma"] else self.default_backend
        
        if current_backend not in self.clients:
            self._init_backend(current_backend)
            
        return self.clients[current_backend]

    async def store_data(self, key: str, value: Any, collection: str = "default", backend: str = None) -> Dict:
        """Stocke des données dans le backend choisi"""
        try:
            client = self._get_client(backend)
            current_backend = backend if backend in ["redis", "mongo", "chroma"] else self.default_backend
            
            if current_backend == "redis":
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                client.set(f"{collection}:{key}", value)
            elif current_backend == "mongo":
                client["db"][collection].update_one(
                    {"_id": key},
                    {"$set": value},
                    upsert=True
                )
            elif current_backend == "chroma":
                collection_obj = client.get_or_create_collection(collection)
                metadata = {}
                if isinstance(value, dict):
                    metadata = {
                        "type": value.get("type", "file"),
                        **(value.get("metadata", {}))
                    }
                collection_obj.upsert(
                    documents=[json.dumps(value)],
                    ids=[key],
                    metadatas=[metadata]
                )
            return {"status": "success", "key": key}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def get_data(self, key: str, collection: str = "default", backend: str = None) -> Any:
        """Récupère des données depuis le backend"""
        try:
            client = self._get_client(backend)
            current_backend = backend if backend in ["redis", "mongo", "chroma"] else self.default_backend
            
            if current_backend == "redis":
                data = client.get(f"{collection}:{key}")
                try:
                    return json.loads(data) if data else None
                except:
                    return data
            elif current_backend == "mongo":
                data = client["db"][collection].find_one({"_id": key})
                return data if data else None
            elif current_backend == "chroma":
                collection_obj = client.get_or_create_collection(collection)
                results = collection_obj.get(ids=[key])
                return json.loads(results['documents'][0]) if results['documents'] else None
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def delete_data(self, key: str, collection: str = "default", backend: str = None) -> Dict:
        """Supprime des données du backend"""
        try:
            client = self._get_client(backend)
            current_backend = backend if backend in ["redis", "mongo", "chroma"] else self.default_backend
            
            if current_backend == "redis":
                client.delete(f"{collection}:{key}")
            elif current_backend == "mongo":
                client["db"][collection].delete_one({"_id": key})
            elif current_backend == "chroma":
                collection_obj = client.get_or_create_collection(collection)
                collection_obj.delete(ids=[key])
            return {"status": "success", "key": key}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def list_data(self, collection: str = "default", pattern: str = "*", backend: str = None) -> List:
        """Liste les données dans une collection"""
        try:
            client = self._get_client(backend)
            current_backend = backend if backend in ["redis", "mongo", "chroma"] else self.default_backend
            
            if current_backend == "redis":
                keys = client.keys(f"{collection}:{pattern}")
                return [key.split(':', 1)[1] for key in keys]
            elif current_backend == "mongo":
                return [doc["_id"] for doc in client["db"][collection].find()]
            elif current_backend == "chroma":
                collection_obj = client.get_or_create_collection(collection)
                return collection_obj.get()["ids"]
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def search_data(self, query: str, collection: str = "default", backend: str = None, **kwargs) -> List:
        """Recherche des données dans le backend"""
        try:
            client = self._get_client(backend)
            current_backend = backend if backend in ["redis", "mongo", "chroma"] else self.default_backend
            
            if current_backend == "redis":
                return {"status": "error", "message": "Search not supported for Redis"}
            elif current_backend == "mongo":
                results = client["db"][collection].find(
                    {"$text": {"$search": query}},
                    limit=kwargs.get('n_results', 10)
                )
                return list(results)
            elif current_backend == "chroma":
                collection_obj = client.get_or_create_collection(collection)
                where = kwargs.get('where')
                results = collection_obj.query(
                    query_texts=[query],
                    n_results=kwargs.get('n_results', 10),
                    where=where
                )
                return [
                    {
                        "id": id,
                        "document": json.loads(doc),
                        "metadata": meta,
                        "distance": distance
                    }
                    for id, doc, meta, distance in zip(
                        results['ids'][0],
                        results['documents'][0],
                        results['metadatas'][0],
                        results['distances'][0]
                    )
                ]
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def list_collections(self, backend: str = None) -> List[str]:
        """Liste toutes les collections disponibles"""
        try:
            client = self._get_client(backend)
            current_backend = backend if backend in ["redis", "mongo", "chroma"] else self.default_backend
            
            if current_backend == "redis":
                keys = client.keys("*")
                collections = set()
                for key in keys:
                    if ":" in key:
                        collections.add(key.split(":")[0])
                return list(collections)
            elif current_backend == "mongo":
                return client["db"].list_collection_names()
            elif current_backend == "chroma":
                return client.list_collections()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def list_databases(self, backend: str = None) -> List[str]:
        """Liste toutes les bases de données disponibles"""
        try:
            client = self._get_client(backend)
            current_backend = backend if backend in ["redis", "mongo", "chroma"] else self.default_backend
            
            if current_backend == "redis":
                return [str(i) for i in range(16)]
            elif current_backend == "mongo":
                return client["client"].list_database_names()
            elif current_backend == "chroma":
                return ["default"]
        except Exception as e:
            return {"status": "error", "message": str(e)}