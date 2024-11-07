from typing import Optional, Any, Dict, List
import redis
import json
from pymongo import MongoClient
import chromadb
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StorageManager:
    def __init__(self, backend: str = "redis", **kwargs):
        """
        Initialise le gestionnaire de stockage
        
        Args:
            backend: Type de backend ("redis", "mongo", "chroma")
            **kwargs: Configuration spécifique au backend
        """
        self.backend = backend
        self.config = kwargs
        logger.info(f"Initializing StorageManager with backend: {backend}")
        logger.info(f"Configuration: {kwargs}")
        self._init_backend()

    def _init_backend(self):
        try:
            if self.backend == "redis":
                redis_host = self.config.get('redis_host', 'localhost')
                redis_port = int(self.config.get('redis_port', 6379))
                logger.info(f"Connecting to Redis at {redis_host}:{redis_port}")
                self.client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    decode_responses=True
                )
                self.client.ping()  # Vérifie la connexion
                logger.info("Successfully connected to Redis")

            elif self.backend == "mongo":
                mongo_url = self.config.get('mongo_url', 'mongodb://localhost:27017')
                mongo_db = self.config.get('mongo_db', 'default')
                logger.info(f"Connecting to MongoDB at {mongo_url}, database: {mongo_db}")
                self.client = MongoClient(mongo_url)
                self.db = self.client[mongo_db]
                # Vérifie la connexion
                self.client.admin.command('ping')
                logger.info("Successfully connected to MongoDB")

            elif self.backend == "chroma":
                persist_dir = self.config.get('chroma_persist_dir', './chroma_storage')
                logger.info(f"Initializing ChromaDB with persist_dir: {persist_dir}")
                self.client = chromadb.PersistentClient(path=persist_dir)
                logger.info("Successfully initialized ChromaDB")

            else:
                raise ValueError(f"Backend non supporté: {self.backend}")

        except Exception as e:
            logger.error(f"Error initializing {self.backend} backend: {str(e)}")
            raise

    async def store_data(self, key: str, value: Any, collection: str = "default") -> Dict:
        """Stocke des données dans le backend choisi"""
        try:
            if self.backend == "redis":
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                self.client.set(f"{collection}:{key}", value)
            elif self.backend == "mongo":
                self.db[collection].update_one(
                    {"_id": key},
                    {"$set": value},
                    upsert=True
                )
            elif self.backend == "chroma":
                collection = self.client.get_or_create_collection(collection)
                collection.upsert(
                    documents=[json.dumps(value)],
                    ids=[key]
                )
            return {"status": "success", "key": key}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def get_data(self, key: str, collection: str = "default") -> Any:
        """Récupère des données depuis le backend"""
        try:
            if self.backend == "redis":
                data = self.client.get(f"{collection}:{key}")
                try:
                    return json.loads(data) if data else None
                except:
                    return data
            elif self.backend == "mongo":
                data = self.db[collection].find_one({"_id": key})
                return data if data else None
            elif self.backend == "chroma":
                collection = self.client.get_or_create_collection(collection)
                results = collection.get(ids=[key])
                return json.loads(results['documents'][0]) if results['documents'] else None
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def delete_data(self, key: str, collection: str = "default") -> Dict:
        """Supprime des données du backend"""
        try:
            if self.backend == "redis":
                self.client.delete(f"{collection}:{key}")
            elif self.backend == "mongo":
                self.db[collection].delete_one({"_id": key})
            elif self.backend == "chroma":
                collection = self.client.get_or_create_collection(collection)
                collection.delete(ids=[key])
            return {"status": "success", "key": key}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def list_data(self, collection: str = "default", pattern: str = "*") -> List:
        """Liste les données disponibles dans le backend"""
        try:
            if self.backend == "redis":
                keys = self.client.keys(f"{collection}:{pattern}")
                return [key.split(':', 1)[1] for key in keys]
            elif self.backend == "mongo":
                return [doc["_id"] for doc in self.db[collection].find({}, {"_id": 1})]
            elif self.backend == "chroma":
                collection = self.client.get_or_create_collection(collection)
                return collection.get()["ids"]
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def search_data(self, query: str, collection: str = "default", **kwargs) -> List:
        """Recherche des données dans le backend"""
        try:
            if self.backend == "chroma":
                collection = self.client.get_or_create_collection(collection)
                results = collection.query(
                    query_texts=[query],
                    n_results=kwargs.get('n_results', 10)
                )
                return [
                    {
                        "id": id,
                        "document": json.loads(doc),
                        "distance": distance
                    }
                    for id, doc, distance in zip(
                        results['ids'][0],
                        results['documents'][0],
                        results['distances'][0]
                    )
                ]
            else:
                return {"status": "error", "message": "Search not supported for this backend"}
        except Exception as e:
            return {"status": "error", "message": str(e)} 