import os
import shutil
import zipfile
import asyncio
import base64
from pathlib import Path
from typing import List, Dict, Optional, AsyncGenerator, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import mimetypes

@dataclass
class StreamResponse:
    type: str  # "progress", "status", "error", "data"
    content: str
    metadata: Optional[Dict[str, Any]] = None

class FileCallbacks:
    def __init__(self,
                 on_progress: Optional[Callable[[int, str], None]] = None,
                 on_complete: Optional[Callable[[str], None]] = None,
                 on_error: Optional[Callable[[Exception], None]] = None,
                 should_stop: Optional[Callable[[], bool]] = None):
        self.on_progress = on_progress
        self.on_complete = on_complete
        self.on_error = on_error
        self.should_stop = should_stop or (lambda: False)

class FileManager:
    def __init__(self, base_path: str = "/app/Shared"):
        self.base_path = Path(base_path)
        self._stop_event = asyncio.Event()
        if not self.base_path.exists():
            self.base_path.mkdir(parents=True, exist_ok=True)

    def stop(self):
        """Arrête l'opération en cours"""
        if not self._stop_event.is_set():
            self._stop_event.set()

    def reset(self):
        """Réinitialise le gestionnaire"""
        self._stop_event.clear()

    def _validate_path(self, path: str) -> Path:
        """Valide et retourne un chemin sécurisé"""
        clean_path = Path(path).resolve().relative_to(Path(path).resolve().anchor)
        full_path = (self.base_path / clean_path).resolve()
        print('full_path',path, full_path, clean_path,Path(path),Path(path).resolve().anchor)
        
        if not str(full_path).startswith(str(self.base_path)):
            raise ValueError("Chemin non autorisé")
        return full_path

    async def create_directory(self, path: str) -> Dict:
        """Crée un nouveau dossier"""
        try:
            dir_path = self._validate_path(path)
            dir_path.mkdir(parents=True, exist_ok=True)
            return {
                "status": "success",
                "message": f"Dossier créé: {path}",
                "timestamp": str(datetime.now())
            }
        except Exception as e:
            return {"error": str(e)}

    async def move(self, source: str, destination: str) -> Dict[str, Any]:
        """
        Déplace un fichier ou un répertoire vers une destination existante
        """
        try:
            source_path = self._validate_path(source)
            dest_path = self._validate_path(destination)

            if not source_path.exists():
                return {"error": f"Le chemin source n'existe pas: {source}"}

            # Vérifier que la destination existe et est un répertoire
            if not dest_path.is_dir():
                return {"error": f"La destination doit être un répertoire existant: {destination}"}

            # Construire le chemin complet de destination
            final_dest = dest_path / source_path.name

            # Vérifier qu'on ne déplace pas un répertoire dans lui-même
            if source_path.is_dir():
                if str(dest_path).startswith(str(source_path)):
                    return {"error": "Impossible de déplacer un répertoire dans lui-même"}

            # Vérifier si la destination finale existe déjà
            if final_dest.exists():
                return {"error": f"Un élément existe déjà à la destination: {final_dest}"}

            # Déplacer le fichier ou le dossier
            shutil.move(str(source_path), str(final_dest))

            return {
                "status": "success",
                "message": f"{'Dossier' if source_path.is_dir() else 'Fichier'} déplacé avec succès",
                "source": source,
                "destination": str(final_dest.relative_to(self.base_path)),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {"error": f"Erreur lors du déplacement: {str(e)}"}

    async def save_file(self, file_path: str, content: bytes) -> Dict:
        """Sauvegarde un fichier"""
        try:
            full_path = self._validate_path(file_path)
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_bytes(content)
            return {
                "status": "success",
                "message": f"Fichier sauvegardé: {file_path}",
                "size": len(content),
                "timestamp": str(datetime.now())
            }
        except Exception as e:
            return {"error": str(e)}

    async def delete_file(self, path: str) -> Dict:
        """Supprime un fichier"""
        try:
            file_path = self._validate_path(path)
            file_path.unlink()
            return {
                "status": "success",
                "message": f"Fichier supprimé: {path}",
                "timestamp": str(datetime.now())
            }
        except Exception as e:
            return {"error": str(e)}

    async def delete_directory(self, path: str) -> Dict:
        """Supprime un dossier et son contenu"""
        try:
            dir_path = self._validate_path(path)
            shutil.rmtree(str(dir_path))
            return {
                "status": "success",
                "message": f"Dossier supprimé: {path}",
                "timestamp": str(datetime.now())
            }
        except Exception as e:
            return {"error": str(e)}

    async def stream_file(self, path: str, chunk_size: int = 8192,
                         callbacks: Optional[FileCallbacks] = None) -> AsyncGenerator[StreamResponse, None]:
        """Stream un fichier en chunks"""
        try:
            file_path = self._validate_path(path)
            file_size = file_path.stat().st_size
            bytes_sent = 0

            with open(file_path, 'rb') as f:
                while chunk := f.read(chunk_size):
                    if self._stop_event.is_set() or (callbacks and callbacks.should_stop()):
                        yield StreamResponse(type="status", content="stopped")
                        return

                    bytes_sent += len(chunk)
                    progress = (bytes_sent / file_size) * 100
                    chunk_base64 = base64.b64encode(chunk).decode()
                    
                    if callbacks and callbacks.on_progress:
                        callbacks.on_progress(progress, f"Streaming {bytes_sent}/{file_size} bytes")

                    yield StreamResponse(
                        type="data",
                        content=chunk_base64,
                        metadata={
                            "progress": progress,
                            "bytes_sent": bytes_sent,
                            "total_bytes": file_size
                        }
                    )
                    await asyncio.sleep(0.01)

            if callbacks and callbacks.on_complete:
                callbacks.on_complete(path)

            yield StreamResponse(
                type="status",
                content="completed",
                metadata={
                    "path": str(path),
                    "size": file_size
                }
            )

        except Exception as e:
            if callbacks and callbacks.on_error:
                callbacks.on_error(e)
            yield StreamResponse(type="error", content=str(e))
        finally:
            self.reset()

    async def compress(self, path: str, zip_name: Optional[str] = None, is_directory: bool = False,
                      callbacks: Optional[FileCallbacks] = None) -> AsyncGenerator[StreamResponse, None]:
        """
        Compresse un fichier ou un dossier en ZIP avec progression
        
        Args:
            path: Chemin du fichier ou dossier à compresser
            zip_name: Nom du fichier ZIP de sortie (optionnel)
            is_directory: True si c'est un dossier, False si c'est un fichier
            callbacks: Callbacks optionnels pour le suivi
        """
        try:
            source_path = self._validate_path(path)
            
            if not zip_name:
                zip_name = f"{path}.zip"
            
            zip_path = self._validate_path(zip_name)
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                if is_directory:
                    # Parcourir tous les dossiers et fichiers
                    for root, dirs, files in os.walk(source_path):
                        # Calculer le chemin relatif pour le zip
                        rel_path = os.path.relpath(root, source_path)
                        
                        # Ajouter le dossier courant (même vide)
                        if rel_path != '.':
                            # Ajouter un slash à la fin pour indiquer que c'est un dossier
                            zip_path_entry = os.path.join(rel_path, '')
                            zipf.write(root, zip_path_entry)
                        
                        # Ajouter tous les fichiers du dossier courant
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.join(rel_path, file)
                            zipf.write(file_path, arcname)
                            
                            if callbacks:
                                await callbacks.on_progress(file, 0)
                else:
                    # Cas d'un fichier unique
                    zipf.write(source_path, os.path.basename(source_path))
                    
                    if callbacks:
                        await callbacks.on_progress(os.path.basename(source_path), 100)
            
            if callbacks and callbacks.on_complete:
                callbacks.on_complete(str(zip_path))

            yield StreamResponse(
                type="status",
                content="completed",
                metadata={
                    "path": str(zip_path),
                    "size": zip_path.stat().st_size
                }
            )

        except Exception as e:
            if callbacks and callbacks.on_error:
                callbacks.on_error(e)
            yield StreamResponse(type="error", content=str(e))
        finally:
            self.reset()

    async def decompress_zip(self, zip_path: str, extract_path: Optional[str] = None) -> Dict:
        """Décompresse un fichier ZIP"""
        try:
            zip_file = self._validate_path(zip_path)
            if extract_path:
                extract_dir = self._validate_path(extract_path)
            else:
                extract_dir = zip_file.parent / zip_file.stem

            with zipfile.ZipFile(str(zip_file), 'r') as zipf:
                zipf.extractall(str(extract_dir))

            return {
                "status": "success",
                "message": f"Fichier décompressé dans: {extract_dir}",
                "timestamp": str(datetime.now())
            }
        except Exception as e:
            return {"error": str(e)}

    async def list_directory(self, path: str = "") -> Dict:
        """Liste le contenu d'un dossier"""
        try:
            dir_path = self._validate_path(path)
            items = []
            
            # Séparer les dossiers et les fichiers
            directories = []
            files = []
            
            for item in dir_path.iterdir():
                item_info = {
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                    "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                }
                
                if item.is_dir():
                    directories.append(item_info)
                else:
                    files.append(item_info)
            
            # Trier les dossiers et fichiers par nom
            directories.sort(key=lambda x: x["name"].lower())
            files.sort(key=lambda x: x["name"].lower())
            
            # Combiner les listes triées
            items = directories + files

            return {
                "status": "success",
                "path": path,
                "items": items,
                "timestamp": str(datetime.now())
            }
        except Exception as e:
            return {"error": str(e)}

    async def rename(self, path: str, new_name: str) -> Dict[str, Any]:
        """
        Renomme un fichier ou un répertoire
        """
        try:
            source_path = self._validate_path(path)
            
            if not source_path.exists():
                return {"error": f"Le chemin n'existe pas: {path}"}
                
            # Vérifier que le nouveau nom est valide
            if '/' in new_name or '\\' in new_name:
                return {"error": "Le nouveau nom ne peut pas contenir de séparateur de chemin"}
                
            # Construire le nouveau chemin
            new_path = source_path.parent / new_name
            
            # Vérifier que la destination n'existe pas déjà
            if new_path.exists():
                return {"error": f"Un élément avec ce nom existe déjà: {new_name}"}
                
            # Renommer le fichier ou le dossier
            source_path.rename(new_path)
            
            return {
                "status": "success",
                "message": f"{'Dossier' if source_path.is_dir() else 'Fichier'} renommé avec succès",
                "old_path": path,
                "new_path": str(new_path.relative_to(self.base_path)),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Erreur lors du renommage: {str(e)}"}

    async def preview(self, path: str) -> Dict[str, Any]:
        """
        Renvoie le contenu d'un fichier pour prévisualisation
        Supporte: texte, images, pdf, audio, vidéo
        """
        try:
            file_path = self._validate_path(path)
            
            if not file_path.exists():
                return {"error": "Fichier non trouvé"}
                
            # Obtenir le type MIME
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if not mime_type:
                mime_type = 'application/octet-stream'
                
            # Limite de taille pour la prévisualisation (ex: 10MB)
            if file_path.stat().st_size > 10 * 1024 * 1024:
                return {"error": "Fichier trop volumineux pour la prévisualisation"}
                
            # Lecture du contenu selon le type
            if mime_type.startswith('text/'):
                # Fichiers texte
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif mime_type.startswith(('image/', 'audio/', 'video/', 'application/pdf')):
                # Fichiers binaires (images, audio, vidéo, pdf)
                with open(file_path, 'rb') as f:
                    content = base64.b64encode(f.read()).decode('utf-8')
            else:
                return {"error": "Type de fichier non supporté pour la prévisualisation"}
                
            return {
                "status": "success",
                "mime_type": mime_type,
                "content": content,
                "size": file_path.stat().st_size,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Erreur lors de la prévisualisation: {str(e)}"}

    async def save_file_base64(self, file_path: str, content: str, mime_type: str) -> Dict:
        """Sauvegarde un fichier à partir de contenu base64"""
        try:
            full_path = self._validate_path(file_path)
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Décodage du contenu base64
            try:
                file_content = base64.b64decode(content)
            except Exception:
                return {"error": "Contenu base64 invalide"}
                
            # Écriture du fichier
            full_path.write_bytes(file_content)
            
            return {
                "status": "success",
                "message": f"Fichier sauvegardé: {file_path}",
                "size": len(file_content),
                "mime_type": mime_type,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}

# Instance globale du gestionnaire de fichiers
file_manager = FileManager() 