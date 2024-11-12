from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from transformers import CLIPProcessor, CLIPModel
import torch
from os import path
from slugify import slugify
import pytesseract
import json
import os
import io
import base64
import asyncio
from Kapweb.huggingface import download_model
from typing import Callable, Optional, Dict, Any, AsyncGenerator
from dataclasses import dataclass
from PIL import Image
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

current_file = path.realpath(__file__)

def load_media_models(file_path):
    print(file_path)
    with open(file_path, 'r') as file:
        return json.load(file)

@dataclass
class StreamResponse:
    type: str  # "text", "status", "error", "progress", "image"
    content: str
    metadata: Optional[Dict[str, Any]] = None

class MediaCallbacks:
    def __init__(self,
                 on_progress: Optional[Callable[[int, str], None]] = None,
                 on_complete: Optional[Callable[[str], None]] = None,
                 on_error: Optional[Callable[[Exception], None]] = None,
                 should_stop: Optional[Callable[[], bool]] = None):
        self.on_progress = on_progress
        self.on_complete = on_complete
        self.on_error = on_error
        self.should_stop = should_stop or (lambda: False)

class MediaGenerator:
    def __init__(self, cache_dir=None, models_config_path=None):
        print(f"Initialisation de MediaGenerator avec models_config_path: {models_config_path} {cache_dir}")
        self.cache_dir = cache_dir or path.join(path.dirname(current_file), "..", "Cache")
        self.device =  "mps"if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float32  #torch.float32 if self.device == "mps" else torch.float16  # float32 pour MPS
        self.pipe = None
        self.current_model = None
        self.models_config_path = models_config_path or path.join(path.dirname(current_file),"..")
        self.model_cache_dir = path.join(self.cache_dir, "LlamaCppModel")
        self._stop_event = asyncio.Event()
        self.progress_queue = Queue()
        self.executor = ThreadPoolExecutor(max_workers=1)
        
    def get_model_config(self, model_type):
        self.models_config = load_media_models( path.join(self.models_config_path, "image_models.json"))
        """Récupère la configuration d'un modèle à partir de son type"""
        parts = model_type.split('/')
        config = self.models_config
        for part in parts:
            if part in config:
                config = config[part]
            else:
                raise ValueError(f"Type de modèle non trouvé: {model_type}")
        return config

    async def init_model(self, model_type="sdxl/turbo"):
        """Initialise le modèle de génération"""
        try:
            print(f"\nInitialisation du modèle {model_type}")
            model_config = self.get_model_config(model_type)
            
            if self.current_model == model_type:
                print("Modèle déjà chargé")
                return

            print(f"Chargement du modèle: {model_config['name']}")
            
            # Vérifier si le modèle est dans le cache
            model_path = path.join(self.model_cache_dir, model_config["model_id"])
            if not path.exists(model_path):
                raise ValueError(f"Le modèle n'est pas téléchargé: {model_type}")
            
            model_kwargs = model_config['config'].copy()
            if self.device == "cpu":
                model_kwargs["torch_dtype"] = self.torch_dtype
                if "variant" in model_kwargs:
                    del model_kwargs["variant"]
            model_cache_path = path.join(self.model_cache_dir, model_config["model_id"])
            if model_config['type'] == 'text2image':
                self.pipe = AutoPipelineForText2Image.from_pretrained(
                    model_config['model_id'],
                    cache_dir=model_cache_path,
                    ignore_mismatched_sizes=True,
                    **model_kwargs
                )
            elif model_config['type'] == 'image2image':
                self.pipe = AutoPipelineForImage2Image.from_pretrained(
                    model_config['model_id'],
                    cache_dir=model_cache_path,
                    ignore_mismatched_sizes=True,
                    **model_kwargs
                )
                
            self.pipe.to(self.device)
            self.current_model = model_type
            print(f"Modèle {model_type} chargé avec succès sur {self.device}")
            
        except Exception as e:
            self.pipe = None
            print(f"Erreur lors du chargement du modèle: {str(e)}")
            raise

    def stop(self):
        """Arrête la génération en cours"""
        if not self._stop_event.is_set():
            self._stop_event.set()

    def reset(self):
        """Réinitialise le générateur"""
        self._stop_event.clear()

    async def generate_image(self, prompt: str, negative_prompt: str = "", 
                           width: int = 1024, height: int = 1024, steps: int = 20):
        try:
            if not self.pipe:
                raise Exception("Modèle non initialisé")

            def callback_on_step_end(pipe, step_index, timestep, callback_kwargs):
                progress = (step_index + 1) / steps * 100
                self.progress_queue.put({"progress": progress})
                return {"progress": progress}

            def run_pipeline():
                return self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    guidance_scale=0.0,
                    num_inference_steps=steps,
                    callback_on_step_end=callback_on_step_end
                )

            future = self.executor.submit(run_pipeline)

            while not future.done():
                try:
                    if not self.progress_queue.empty():
                        progress = self.progress_queue.get_nowait()
                        yield progress
                    await asyncio.sleep(0.1)
                except Exception as e:
                    yield {"error": f"Erreur de progression: {str(e)}"}

            result = future.result()
            image = result.images[0]

            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            yield {"status": "completed", "image": f"{img_str}"}
            
        except Exception as e:
            yield {"error": str(e)}

    async def refine_image_data(self, image: Image, prompt: str, negative_prompt: str = "",
                               strength: float = 0.3, steps: int = 20):
        try:
            if not self.pipe:
                raise Exception("Modèle non initialisé")

            yield {"status": "starting", "prompt": prompt}
            
            def callback_on_step_end(pipe, step_index, timestep, callback_kwargs):
                progress = (step_index + 1) / steps * 100
                self.progress_queue.put({"progress": progress})
                return {"progress": progress}

            def run_pipeline():
                return self.pipe(
                    image=image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    strength=strength,
                    num_inference_steps=steps,
                    callback_on_step_end=callback_on_step_end
                )

            future = self.executor.submit(run_pipeline)

            while not future.done():
                try:
                    if not self.progress_queue.empty():
                        progress = self.progress_queue.get_nowait()
                        yield progress
                    await asyncio.sleep(0.1)
                except Exception as e:
                    yield {"error": f"Erreur de progression: {str(e)}"}

            result = future.result()
            refined = result.images[0]

            buffered = io.BytesIO()
            refined.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            yield {"status": "completed", "image": f"data:image/png;base64,{img_str}"}
            
        except Exception as e:
            yield {"error": str(e)}

    def get_available_models(self):
        self.models_config = load_media_models( path.join(self.models_config_path, "image_models.json"))
        """Retourne la liste des modèles disponibles avec leurs détails"""
        models = {}
        for category, category_models in self.models_config.items():
            for model_name, model_config in category_models.items():
                model_id = f"{category}/{model_name}"
                models[model_id] = {
                    "name": model_config["name"],
                    "type": model_config["type"]
                }
        return models

    def save_image(self, image, prompt, output_dir="Outputs"):
        """Sauvegarde l'image générée"""
        dirpath = path.join(path.dirname(current_file), "..", output_dir)
        os.makedirs(dirpath, exist_ok=True)
        
        image_name = f'{slugify(prompt)}.png'
        image_path = path.join(dirpath, image_name)
        image.save(image_path)
        return image_name

    async def load_model(self, model_type):
        """Télécharge et charge le modèle si nécessaire"""
        try:
            model_config = self.get_model_config(model_type)
            model_info = {
                "model_name": model_config["model_id"],
                "model_file": model_config.get("model_file")
            }
            print(model_info)
            # Vérifier si le modèle doit être téléchargé
            async for chunk in download_model(model_info):
                yield chunk

            # Initialiser le modèle
            await self.init_model(model_type)
            
        except Exception as e:
            yield f'data: {{"error": "Erreur lors du chargement du modèle: {str(e)}"}}\n\n'

    def __del__(self):
        self.executor.shutdown(wait=False)

class MediaAnalyzer:
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or path.join(path.dirname(current_file), "..", "Cache")
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    def ocr_image_data(self, image):
        """Extrait le texte d'une image fournie en données"""
        result = pytesseract.image_to_string(image)
        return result

    async def analyze_image_data(self, image, labels=[], model_name="laion/CLIP-ViT-H-14-laion2B-s32B-b79K"):
        """Analyse une image fournie en données avec CLIP"""
        try:
            image = image.resize((512, 512))
            model = CLIPModel.from_pretrained(model_name)
            processor = CLIPProcessor.from_pretrained(model_name)

            inputs = processor(text=labels, images=[image], return_tensors="pt", padding=True)
            outputs = model(**inputs)
            
            logits = outputs.logits_per_image
            probs = logits.softmax(dim=1)
            
            return {
                "logits": logits.tolist(),
                "probabilities": probs.tolist()
            }

        except Exception as e:
            return {"error": str(e)}

# Instance globale du générateur de médias
media_generator = MediaGenerator()
media_analyzer = MediaAnalyzer() 