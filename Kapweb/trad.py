from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from os import path
import torch
from Kapweb.huggingface import download_model
from typing import Callable, Optional, Dict, Any, AsyncGenerator
from dataclasses import dataclass
import asyncio

current_file = path.realpath(__file__)

@dataclass
class StreamResponse:
    type: str  # "text", "status", "error", "progress"
    content: str
    metadata: Optional[Dict[str, Any]] = None

class TranslationCallbacks:
    def __init__(self,
                 on_progress: Optional[Callable[[int, str], None]] = None,
                 on_complete: Optional[Callable[[str], None]] = None,
                 on_error: Optional[Callable[[Exception], None]] = None,
                 should_stop: Optional[Callable[[], bool]] = None):
        self.on_progress = on_progress
        self.on_complete = on_complete
        self.on_error = on_error
        self.should_stop = should_stop or (lambda: False)

class Translator:
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or path.join(path.dirname(current_file), "..", "Cache")
        self.models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._is_ready = True
        self._stop_event = asyncio.Event()
    
    def stop(self):
        """Arrête la traduction en cours"""
        if not self._stop_event.is_set():
            self._stop_event.set()

    def reset(self):
        """Réinitialise le traducteur"""
        self._stop_event.clear()

    def is_ready(self):
        """Vérifie si le traducteur est prêt"""
        return self._is_ready
    
    def get_model_name(self, from_lang, to_lang):
        return f"Helsinki-NLP/opus-mt-{from_lang}-{to_lang}"
    
    async def load_model(self, from_lang, to_lang, callbacks: Optional[TranslationCallbacks] = None):
        """Télécharge et charge le modèle si nécessaire"""
        try:
            model_name = self.get_model_name(from_lang, to_lang)
            model_info = {
                "model_name": model_name,
                "model_file": "pytorch_model.bin"
            }
            
            if callbacks and callbacks.on_progress:
                callbacks.on_progress(0, "Téléchargement du modèle...")
            yield StreamResponse(type="progress", content="0", metadata={"message": "Téléchargement..."})

            # Téléchargement du modèle
            async for chunk in download_model(model_info):
                yield StreamResponse(type="status", content="downloading", metadata={"chunk": chunk})

            # Chargement du modèle
            model_key = f"{from_lang}-{to_lang}"
            if model_key not in self.models:
                if callbacks and callbacks.on_progress:
                    callbacks.on_progress(50, "Chargement du modèle...")
                yield StreamResponse(type="progress", content="50", metadata={"message": "Chargement..."})

                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    cache_dir=path.join(self.cache_dir, "TranslationModels")
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    cache_dir=path.join(self.cache_dir, "TranslationModels")
                ).to(self.device)
                self.models[model_key] = (tokenizer, model)
                
            if callbacks and callbacks.on_complete:
                callbacks.on_complete(model_name)
            yield StreamResponse(type="status", content="loaded", metadata={"model": model_name})
            
        except Exception as e:
            if callbacks and callbacks.on_error:
                callbacks.on_error(e)
            yield StreamResponse(type="error", content=str(e))
    
    async def translate(self, text, from_lang, to_lang, callbacks: Optional[TranslationCallbacks] = None):
        """Traduit un texte d'une langue à une autre"""
        try:
            model_key = f"{from_lang}-{to_lang}"
            
            # Vérifier si le modèle est chargé
            if model_key not in self.models:
                async for response in self.load_model(from_lang, to_lang, callbacks):
                    yield response
        
            if self._stop_event.is_set() or (callbacks and callbacks.should_stop()):
                yield StreamResponse(type="status", content="stopped")
                return

            tokenizer, model = self.models[model_key]
            
            # Traduction
            inputs = tokenizer(text, return_tensors="pt", padding=True).to(self.device)
            outputs = model.generate(**inputs)
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if callbacks and callbacks.on_complete:
                callbacks.on_complete(translation)

            yield StreamResponse(
                type="status", 
                content="completed",
                metadata={
                    "translated_text": translation,
                    "from": from_lang,
                    "to": to_lang,
                    "model": self.get_model_name(from_lang, to_lang)
                }
            )
            
        except Exception as e:
            print(f"Erreur de traduction: {str(e)}")
            if callbacks and callbacks.on_error:
                callbacks.on_error(e)
            yield StreamResponse(type="error", content=str(e))
        finally:
            self.reset()

    def get_available_languages(self):
        """Liste des paires de langues disponibles"""
        return {
            "en": ["fr", "es", "de", "it"],
            "fr": ["en", "es", "de", "it"],
            "es": ["en", "fr", "de", "it"],
            "de": ["en", "fr", "es", "it"],
            "it": ["en", "fr", "es", "de"]
        }

    def is_valid_language_pair(self, from_lang, to_lang):
        """Vérifie si la paire de langues est valide"""
        languages = self.get_available_languages()
        return from_lang in languages and to_lang in languages[from_lang]

# Instance globale du traducteur
translator = Translator()