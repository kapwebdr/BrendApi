from datetime import datetime
from typing import Dict, List, Optional, Any
from Kapweb.storage import StorageManager
import uuid
import os

class HistoryManager:
    def __init__(self, storage_manager: StorageManager = None, backend: str = None):
        """
        Initialise le gestionnaire d'historique
        
        Args:
            storage_manager: Instance de StorageManager existante (optionnel)
            backend: Type de backend à utiliser ("redis", "mongo", "chroma")
        """
        if storage_manager:
            self.storage = storage_manager
        else:
            self.storage = StorageManager(
                backend=backend or os.getenv('STORAGE_BACKEND', 'mongo'),
                mongo_url=os.getenv('MONGO_URL', 'mongodb://localhost:27017'),
                mongo_db=os.getenv('MONGO_DB', 'default'),
                redis_host=os.getenv('REDIS_HOST', 'localhost'),
                redis_port=int(os.getenv('REDIS_PORT', 6379)),
                chroma_persist_dir=os.getenv('CHROMA_PERSIST_DIR', './chroma_storage')
            )

    async def save_conversation(self, 
        session_id: str,
        id: Optional[str] = None,
        role: str = None,
        message: Any = None,
        metadata: Dict[str, Any] = None
    ) -> Dict:
        """
        Sauvegarde un message dans une conversation
        
        Args:
            session_id: ID de la session
            id: ID de la conversation (généré si non fourni)
            role: Rôle de l'émetteur (user, assistant, system, etc.)
            message: Contenu du message (peut être un objet complexe)
            metadata: Métadonnées additionnelles (modèle utilisé, configuration, etc.)
        """
        try:
            timestamp = datetime.now().isoformat()
            
            # Créer ou récupérer la conversation
            if id:
                conversation = await self.storage.get_data(
                    key=id,
                    collection="conversations"
                )
                if not conversation:
                    return {"status": "error", "message": "Conversation non trouvée"}
            else:
                id = str(uuid.uuid4())
                conversation = {
                    "id": id,
                    "session_id": session_id,
                    "title": str(message)[:50],  # Premier message comme titre
                    "created_at": timestamp,
                    "messages": []
                }

            # Créer le nouveau message
            new_message = {
                "timestamp": timestamp,
                "role": role,
                "message": message,
                "metadata": metadata or {}
            }

            # Ajouter le message à la conversation
            if "messages" not in conversation:
                conversation["messages"] = []
            conversation["messages"].append(new_message)

            # Sauvegarder la conversation mise à jour
            result = await self.storage.store_data(
                key=id,
                value=conversation,
                collection="conversations"
            )

            if result.get("status") == "error":
                return result

            return {
                "status": "success",
                "id": id,
                "message_id": timestamp,
                "title": conversation["title"]
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def get_conversation(self, id: str) -> Dict:
        """
        Récupère une conversation complète
        
        Args:
            id: ID de la conversation
        """
        try:
            conversation = await self.storage.get_data(
                key=id,
                collection="conversations"
            )
            return conversation or {"status": "error", "message": "Conversation non trouvée"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def delete_conversation(self, id: str) -> Dict:
        """
        Supprime une conversation complète
        
        Args:
            id: ID de la conversation
        """
        try:
            result = await self.storage.delete_data(
                key=id,
                collection="conversations"
            )
            return {
                "status": "success",
                "message": f"Conversation {id} supprimée"
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def list_conversations(self, session_id: str) -> List[Dict]:
        """
        Liste les conversations d'une session, triées par date de création
        
        Args:
            session_id: ID de la session
        Returns:
            Liste des conversations triées par date de création (plus récente en premier)
        """
        try:
            conversations = await self.storage.list_data(collection="conversations")
            result = []

            for conv_id in conversations:
                conversation = await self.storage.get_data(
                    key=conv_id,
                    collection="conversations"
                )
                if conversation and conversation.get('session_id') == session_id:
                    result.append({
                        "id": conversation["id"],
                        "title": conversation["title"],
                        "created_at": conversation["created_at"]
                    })

            # Tri par date de création (plus récente en premier)
            return sorted(result, key=lambda x: x["created_at"], reverse=True)

        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def delete_by_session(self, session_id: str) -> Dict:
        """
        Supprime toutes les conversations d'une session
        
        Args:
            session_id: ID de la session
        """
        try:
            conversations = await self.list_conversations(session_id)
            deleted_count = 0
            
            for conv in conversations:
                result = await self.delete_conversation(conv["id"])
                if result.get("status") == "success":
                    deleted_count += 1

            return {
                "status": "success",
                "message": f"Session {session_id} supprimée",
                "deleted_conversations": deleted_count
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

# Instance globale
history_manager = HistoryManager(backend=os.getenv('STORAGE_BACKEND', 'mongo'))