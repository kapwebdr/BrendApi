import os
import shutil
import zipfile
import asyncio
import base64
from pathlib import Path
from typing import List, Dict, Optional, AsyncGenerator
from datetime import datetime

class FileManager:
    def __init__(self, base_path: str = "/app/Shared"):
        self.base_path = Path(base_path)
        if not self.base_path.exists():
            self.base_path.mkdir(parents=True, exist_ok=True)

    def _validate_path(self, path: str) -> Path:
        """Valide et retourne un chemin sécurisé"""
        clean_path = Path(path).resolve().relative_to(Path(path).resolve().anchor)
        full_path = (self.base_path / clean_path).resolve()
        print('full_path', full_path)
        
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

    async def move_directory(self, source: str, destination: str) -> Dict:
        """Déplace un dossier"""
        try:
            src_path = self._validate_path(source)
            dst_path = self._validate_path(destination)
            shutil.move(str(src_path), str(dst_path))
            return {
                "status": "success",
                "message": f"Dossier déplacé de {source} vers {destination}",
                "timestamp": str(datetime.now())
            }
        except Exception as e:
            return {"error": str(e)}

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

    async def move_file(self, source: str, destination: str) -> Dict:
        """Déplace un fichier"""
        try:
            src_path = self._validate_path(source)
            dst_path = self._validate_path(destination)
            shutil.move(str(src_path), str(dst_path))
            return {
                "status": "success",
                "message": f"Fichier déplacé de {source} vers {destination}",
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

    async def stream_file(self, path: str, chunk_size: int = 8192) -> AsyncGenerator[str, None]:
        """Stream un fichier en chunks"""
        try:
            file_path = self._validate_path(path)
            file_size = file_path.stat().st_size
            bytes_sent = 0

            with open(file_path, 'rb') as f:
                while chunk := f.read(chunk_size):
                    bytes_sent += len(chunk)
                    progress = (bytes_sent / file_size) * 100
                    chunk_base64 = base64.b64encode(chunk).decode()
                    
                    yield f'data: {{"progress": {progress:.2f}, "chunk": "{chunk_base64}"}}\n\n'
                    await asyncio.sleep(0.01)

            yield f'data: {{"status": "completed"}}\n\n'

        except Exception as e:
            yield f'data: {{"error": "{str(e)}"}}\n\n'

    async def compress_directory(self, path: str, zip_name: Optional[str] = None) -> Dict:
        """Compresse un dossier en ZIP"""
        try:
            dir_path = self._validate_path(path)
            if not zip_name:
                zip_name = f"{dir_path.name}.zip"
            zip_path = self._validate_path(zip_name)

            with zipfile.ZipFile(str(zip_path), 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(str(dir_path)):
                    for file in files:
                        file_path = Path(root) / file
                        arc_name = file_path.relative_to(dir_path)
                        zipf.write(str(file_path), str(arc_name))

            return {
                "status": "success",
                "message": f"Dossier compressé: {zip_name}",
                "size": zip_path.stat().st_size,
                "timestamp": str(datetime.now())
            }
        except Exception as e:
            return {"error": str(e)}

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
            
            for item in dir_path.iterdir():
                item_info = {
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                    "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                }
                items.append(item_info)

            return {
                "status": "success",
                "path": path,
                "items": items,
                "timestamp": str(datetime.now())
            }
        except Exception as e:
            return {"error": str(e)}

# Instance globale du gestionnaire de fichiers
file_manager = FileManager() 