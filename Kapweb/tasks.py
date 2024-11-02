from typing import Dict, Any
import asyncio
from os import path
from Kapweb.llm import loadLlm, load_models
from Kapweb.huggingface import download_model

class ModelLoadingTask:
    """Gestionnaire de tâche pour le chargement des modèles"""
    
    def __init__(self):
        self.current_file = path.realpath(__file__)

    async def handle_download_progress(self, model: Dict[str, Any], session_task: Dict[str, Any]) -> bool:
        """Gère la progression du téléchargement du modèle"""
        async for progress in download_model(model):
            session_task["progress"] = progress
            if "error" in progress:
                session_task["status"] = "error"
                session_task["error"] = progress["error"]
                return False
            if "status" in progress and progress["status"] == "completed":
                return True
        return False

    async def execute(self, model: Dict[str, Any], session_task: Dict[str, Any]) -> None:
        """Exécute la tâche de chargement du modèle"""
        try:
            # Vérifier si le modèle existe déjà
            model_path = path.join(path.dirname(self.current_file), "..", "Cache", "LlamaCppModel", 
                                 model['model_name'].replace('/', path.sep), model['model_file'])
            
            if not path.exists(model_path):
                # Télécharger le modèle si nécessaire
                download_success = await self.handle_download_progress(model, session_task)
                if not download_success:
                    return

            # Charger le modèle en utilisant la fonction de llm.py
            session_task["progress"] = "Chargement du modèle en mémoire..."
            llm = loadLlm(model)
            
            if llm is None:
                session_task["status"] = "error"
                session_task["error"] = "Échec du chargement du modèle"
                return
                
            session_task["llm"] = llm
            session_task["status"] = "loaded"
            
        except Exception as e:
            session_task["status"] = "error"
            session_task["error"] = str(e)

async def load_model_task(model: Dict[str, Any], session_task: Dict[str, Any]) -> None:
    """Point d'entrée pour le chargement du modèle"""
    task_manager = ModelLoadingTask()
    await task_manager.execute(model, session_task)