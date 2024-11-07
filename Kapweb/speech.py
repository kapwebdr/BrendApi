from TTS.api import TTS
from faster_whisper import WhisperModel
import torch
from os import path
import asyncio
from typing import Callable, Optional, Dict, Any, AsyncGenerator
from dataclasses import dataclass
import numpy as np
import soundfile as sf
import io
import json

@dataclass
class StreamResponse:
    type: str  # "text", "audio", "status", "error", "progress"
    content: str
    metadata: Optional[Dict[str, Any]] = None
current_file = path.realpath(__file__)

class SpeechCallbacks:
    def __init__(self,
                 on_progress: Optional[Callable[[int, str], None]] = None,
                 on_complete: Optional[Callable[[str], None]] = None,
                 on_error: Optional[Callable[[Exception], None]] = None,
                 on_segment: Optional[Callable[[str], None]] = None,
                 should_stop: Optional[Callable[[], bool]] = None):
        self.on_progress = on_progress
        self.on_complete = on_complete
        self.on_error = on_error
        self.on_segment = on_segment
        self.should_stop = should_stop or (lambda: False)

def load_models_config():
    """Charge les configurations des modèles depuis les fichiers JSON"""
    tts_config_path = '/app/Config/tts_models.json'
    stt_config_path = '/app/Config/stt_models.json'
    
    with open(tts_config_path, 'r') as f:
        tts_models = json.load(f)
    with open(stt_config_path, 'r') as f:
        stt_models = json.load(f)
        
    return tts_models, stt_models

class SpeechProcessor:
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or path.join(path.dirname(current_file), "..", "Cache")
        self.stt_model = None
        self.tts_model = None
        self.current_model_size = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self._stop_event = asyncio.Event()
        
        # Chargement des configurations
        self.available_tts_models, self.available_stt_models = load_models_config()

    def stop(self):
        """Arrête le traitement en cours"""
        if not self._stop_event.is_set():
            self._stop_event.set()

    def reset(self):
        """Réinitialise le processeur"""
        self._stop_event.clear()

    def get_available_models(self):
        """Retourne la liste des modèles disponibles"""
        stt_models = list(self.available_stt_models['whisper'].keys())  # ["tiny", "base", "small", "medium", "large"]
        tts_models = self.available_tts_models
        
        return {
            "stt": stt_models,
            "tts": tts_models
        }

    def get_voice_config(self, voice: str, language: str = "fr") -> Dict:
        """Récupère la configuration d'une voix"""
        if language in self.available_tts_models and voice in self.available_tts_models[language]:
            return self.available_tts_models[voice] 
        raise ValueError(f"Voix '{voice}' non disponible pour la langue '{language}'")

    async def init_stt(self, model_size: str = "small"):
        """Initialise le modèle STT"""
        if model_size not in self.available_stt_models['whisper']:
            raise ValueError(f"Taille de modèle '{model_size}' non disponible")
            
        if self.stt_model is None or self.current_model_size != model_size:
            self.stt_model = WhisperModel(
                model_size_or_path=model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=path.join(self.cache_dir, "WhisperCache")
            )
            self.current_model_size = model_size

    async def init_tts(self, voice: str = "elise", language: str = "fr"):
        """Initialise le modèle TTS"""
        voice_config = self.get_voice_config(voice, language)
        if self.tts_model is None:
            self.tts_model = TTS(voice_config["model"]).to(self.device)

    async def speech_to_text(self, audio_data: bytes, model_size: str = "small",
                            callbacks: Optional[SpeechCallbacks] = None) -> AsyncGenerator[StreamResponse, None]:
        try:
            await self.init_stt(model_size)
            
            if callbacks and callbacks.on_progress:
                callbacks.on_progress(0, "Démarrage de la transcription...")
            yield StreamResponse(type="progress", content="0", metadata={"message": "Démarrage..."})

            if self._stop_event.is_set() or (callbacks and callbacks.should_stop()):
                yield StreamResponse(type="status", content="stopped")
                return

            # Conversion des données audio
            audio_buffer = io.BytesIO(audio_data)
            audio_array, sample_rate = sf.read(audio_buffer)
            
            temp_buffer = io.BytesIO()
            sf.write(temp_buffer, audio_array, sample_rate, format='WAV')
            temp_buffer.seek(0)
     
            # Transcription avec faster-whisper
            segments_iterator, info = self.stt_model.transcribe(
                temp_buffer,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            full_text = ""
            for segment in segments_iterator:
                if self._stop_event.is_set() or (callbacks and callbacks.should_stop()):
                    yield StreamResponse(type="status", content="stopped")
                    return
                
                segment_text = segment.text.strip()
                full_text += segment_text + " "
                
                # Envoi du segment
                if callbacks and callbacks.on_segment:
                    callbacks.on_segment(segment_text)
                yield StreamResponse(
                    type="segment", 
                    content=segment_text,
                    metadata={
                        "start": segment.start,
                        "end": segment.end
                    }
                )
            
            if callbacks and callbacks.on_complete:
                callbacks.on_complete(full_text.strip())

            yield StreamResponse(type="text", content=full_text.strip())
            yield StreamResponse(type="status", content="completed", metadata={"text": full_text.strip()})

        except Exception as e:
            if callbacks and callbacks.on_error:
                callbacks.on_error(e)
            yield StreamResponse(type="error", content=str(e))
        finally:
            self.reset()

    async def text_to_speech(self, text: str, voice: str = "elise", language: str = "fr",
                            callbacks: Optional[SpeechCallbacks] = None) -> AsyncGenerator[StreamResponse, None]:
        try:
            await self.init_tts(voice, language)
            
            if callbacks and callbacks.on_progress:
                callbacks.on_progress(0, "Démarrage de la synthèse...")
            yield StreamResponse(type="progress", content="0", metadata={"message": "Démarrage..."})

            if self._stop_event.is_set() or (callbacks and callbacks.should_stop()):
                yield StreamResponse(type="status", content="stopped")
                return

            wav = self.tts_model.tts(text=text)

            if callbacks and callbacks.on_complete:
                callbacks.on_complete(wav)

            yield StreamResponse(type="audio", content=wav)
            yield StreamResponse(type="status", content="completed", metadata={"audio": wav})

        except Exception as e:
            if callbacks and callbacks.on_error:
                callbacks.on_error(e)
            yield StreamResponse(type="error", content=str(e))
        finally:
            self.reset()

# Instance globale
speech_processor = SpeechProcessor()