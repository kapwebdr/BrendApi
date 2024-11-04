from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from os import path
import torch
from Kapweb.huggingface import download_model

current_file = path.realpath(__file__)

class Translator:
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or path.join(path.dirname(current_file), "..", "Cache")
        self.models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._is_ready = True
    
    def is_ready(self):
        """Vérifie si le traducteur est prêt"""
        return self._is_ready
    
    def get_model_name(self, from_lang, to_lang):
        return f"Helsinki-NLP/opus-mt-{from_lang}-{to_lang}"
    
    async def load_model(self, from_lang, to_lang):
        """Télécharge et charge le modèle si nécessaire"""
        try:
            model_name = self.get_model_name(from_lang, to_lang)
            model_info = {
                "model_name": model_name,
                "model_file": "pytorch_model.bin"
            }
            
            # Téléchargement du modèle
            async for chunk in download_model(model_info):
                yield chunk

            # Chargement du modèle
            model_key = f"{from_lang}-{to_lang}"
            if model_key not in self.models:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    cache_dir=path.join(self.cache_dir, "TranslationModels")
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    cache_dir=path.join(self.cache_dir, "TranslationModels")
                ).to(self.device)
                self.models[model_key] = (tokenizer, model)
                
            yield f'data: {{"status": "loaded", "model": "{model_name}"}}\n\n'
            
        except Exception as e:
            yield f'data: {{"error": "Erreur lors du chargement du modèle: {str(e)}"}}\n\n'
    
    async def translate(self, text, from_lang, to_lang):
        """Traduit un texte d'une langue à une autre"""
        try:
            model_key = f"{from_lang}-{to_lang}"
            
            # Vérifier si le modèle est chargé
            if model_key not in self.models:
                async for chunk in self.load_model(from_lang, to_lang):
                    yield chunk
        
            tokenizer, model = self.models[model_key]
            
            # Traduction
            inputs = tokenizer(text, return_tensors="pt", padding=True).to(self.device)
            outputs = model.generate(**inputs)
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            yield f'data: {{"status": "completed","translated_text": "{translation}","from": "{from_lang}","to": "{to_lang}","model": "{self.get_model_name(from_lang, to_lang)}"}}\n\n'
            
        except Exception as e:
            print(f"Erreur de traduction: {str(e)}")
            yield f'data: {{"error": "{str(e)}"}}\n\n'

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