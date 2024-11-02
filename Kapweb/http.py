import requests
from bs4 import BeautifulSoup
import asyncio
from urllib.parse import urlparse
from pytube import YouTube
import aiohttp
import io
import base64
from typing import List, Dict, Optional, Generator, AsyncGenerator
from enum import Enum

class ContentType(str, Enum):
    HTML = "html"
    JSON = "json"
    VIDEO = "video"
    IMAGE = "image"
    AUDIO = "audio"
    TEXT = "text"
    BINARY = "binary"

class HTTPProcessor:
    def __init__(self):
        self.session = None
        self.chunk_size = 1024 * 8  # 8KB chunks pour le streaming

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _detect_content_type(self, headers: Dict) -> ContentType:
        """Détecte le type de contenu basé sur les headers"""
        content_type = headers.get('Content-Type', '').lower()
        
        if 'text/html' in content_type:
            return ContentType.HTML
        elif 'application/json' in content_type:
            return ContentType.JSON
        elif 'video' in content_type:
            return ContentType.VIDEO
        elif 'image' in content_type:
            return ContentType.IMAGE
        elif 'audio' in content_type:
            return ContentType.AUDIO
        elif 'text' in content_type:
            return ContentType.TEXT
        else:
            return ContentType.BINARY

    async def stream_content(self, url: str) -> AsyncGenerator[str, None]:
        """Stream le contenu d'une URL avec progression et encodage base64"""
        try:
            async with self.session.get(url) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0
                
                async for chunk in response.content.iter_chunked(self.chunk_size):
                    chunk_base64 = base64.b64encode(chunk).decode()
                    downloaded += len(chunk)
                    progress = (downloaded / total_size * 100) if total_size > 0 else 0
                    
                    yield f'data: {{"progress": {progress:.2f}, "chunk": "{chunk_base64}", "content_type": "{self._detect_content_type(response.headers)}"}}\n\n'
                
                yield 'data: {"status": "completed"}\n\n'
                
        except Exception as e:
            yield f'data: {{"error": "Erreur lors du streaming : {str(e)}"}}\n\n'

    async def extract_content(self, url: str, selectors: Optional[List[str]] = None) -> AsyncGenerator[str, None]:
        """Extrait le contenu HTML basé sur les sélecteurs CSS"""
        try:
            async with self.session.get(url) as response:
                content_type = self._detect_content_type(response.headers)
                
                if content_type == ContentType.HTML:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    if selectors:
                        for selector in selectors:
                            elements = soup.select(selector)
                            for element in elements:
                                yield f'data: {{"selector": "{selector}", "content": "{element.get_text(strip=True)}"}}\n\n'
                    else:
                        yield f'data: {{"content": "{soup.get_text(strip=True)}"}}\n\n'
                else:
                    yield f'data: {{"error": "Le contenu n\'est pas au format HTML"}}\n\n'
                    
        except Exception as e:
            yield f'data: {{"error": "Erreur lors de l\'extraction : {str(e)}"}}\n\n'

    async def stream_youtube(self, video_id: str) -> AsyncGenerator[str, None]:
        """Stream une vidéo YouTube"""
        try:
            url = f'https://www.youtube.com/watch?v={video_id}'
            yt = YouTube(url)
            
            # Info de la vidéo
            yield f'data: {{"title": "{yt.title}", "author": "{yt.author}", "length": {yt.length}}}\n\n'
            
            # Sélectionner le meilleur stream progressif
            stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            
            # Stream le contenu
            async for chunk in self.stream_content(stream.url):
                yield chunk

        except Exception as e:
            yield f'data: {{"error": "Erreur lors du streaming YouTube : {str(e)}"}}\n\n'

    async def analyze_url(self, url: str) -> Dict:
        """Analyse une URL et retourne ses métadonnées"""
        try:
            async with self.session.get(url) as response:
                content_type = self._detect_content_type(response.headers)
                parsed_url = urlparse(url)
                
                return {
                    "url": url,
                    "domain": parsed_url.netloc,
                    "path": parsed_url.path,
                    "content_type": content_type,
                    "size": response.headers.get('Content-Length'),
                    "headers": dict(response.headers),
                    "status": response.status
                }
        except Exception as e:
            return {"error": f"Erreur lors de l'analyse : {str(e)}"}

# Instance globale du processeur HTTP
http_processor = HTTPProcessor()