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
    type: str  # "text", "audio", "status", "error", "progress", "segment"
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

class SpeechProcessor:
    def __init__(self, cache_dir=None):
        # Utilise le cache_dir fourni ou le chemin par défaut
        self.cache_dir = cache_dir or path.join(path.dirname(current_file), "..", "Cache")
        print(f"Initialisation SpeechProcessor avec cache_dir: {self.cache_dir}")
        
        self.stt_model = None
        self.tts_model = None
        self.current_model_size = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self._stop_event = asyncio.Event()
        
        # Chargement des configurations
        self.load_models_config()

    def load_models_config(self):
        """Charge les configurations des modèles depuis les fichiers JSON"""
        tts_config_path = '/app/Config/tts_models.json'
        stt_config_path = '/app/Config/stt_models.json'
        
        with open(tts_config_path, 'r') as f:
            self.tts_config = json.load(f)
        with open(stt_config_path, 'r') as f:
            self.stt_config = json.load(f)

    def stop(self):
        """Arrête le traitement en cours"""
        if not self._stop_event.is_set():
            self._stop_event.set()

    def reset(self):
        """Réinitialise le processeur"""
        self._stop_event.clear()

    def get_available_models(self):
        """Retourne la liste des modèles disponibles"""
        return {
            "stt": self.stt_config,
            "tts": self.tts_config
        }

    async def init_stt(self, model_size: str = "small"):
        """Initialise le modèle STT"""
        if model_size not in self.stt_config["whisper"]["models"]:
            raise ValueError(f"Taille de modèle '{model_size}' non disponible")
            
        model_config = self.stt_config["whisper"]["models"][model_size]
        
        if self.stt_model is None or self.current_model_size != model_size:
            whisper_cache = path.join(self.cache_dir, "WhisperCache")
            print(f"Chargement du modèle Whisper depuis: {whisper_cache}")
            
            self.stt_model = WhisperModel(
                model_size_or_path=model_size,
                device=self.device,
                compute_type=model_config["compute_type"],
                download_root=whisper_cache
            )
            self.current_model_size = model_size

    async def init_tts(self, model_type: str = "xtts_v2", voice: str = None, language: str = None):
        """Initialise le modèle TTS"""
        if model_type not in self.tts_config:
            raise ValueError(f"Type de modèle '{model_type}' non disponible")
            
        model_config = self.tts_config[model_type]
        
        if model_config["requires_language"] and not language:
            raise ValueError(f"Le modèle {model_type} nécessite une langue")
            
        if model_config["type"] == "multi_speaker" and not voice:
            raise ValueError(f"Le modèle {model_type} nécessite une voix")
        
        # Configuration spécifique selon le type de modèle
        if model_config["type"] == "multi_speaker":
            voice_config = next((v for v in model_config["voices"] if v["path"] == voice), None)
            if not voice_config:
                raise ValueError(f"Voix '{voice}' non disponible")
            
            tts_cache = path.join(self.cache_dir, "TTSCache")
            print(f"Chargement du modèle TTS depuis: {tts_cache}")
            self.tts_model = TTS(
                model_config["model_path"],
                progress_bar=True,
                # cache_dir=tts_cache
            ).to(self.device)

    async def speech_to_text(self, audio_data: bytes, model_size: str = "small",
                            callbacks: Optional[SpeechCallbacks] = None) -> AsyncGenerator[StreamResponse, None]:
        try:
            await self.init_stt(model_size)
            model_config = self.stt_config["whisper"]["models"][model_size]
            
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
                beam_size=model_config["beam_size"],
                vad_filter=model_config["vad_filter"],
                vad_parameters=model_config["vad_parameters"]
            )
            
            full_text = ""
            for segment in segments_iterator:
                if self._stop_event.is_set() or (callbacks and callbacks.should_stop()):
                    yield StreamResponse(type="status", content="stopped")
                    return
                
                segment_text = segment.text.strip()
                full_text += segment_text + " "
                
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

    async def text_to_speech(self, text: str, model_type: str = "xtts_v2", voice: str = "elise", 
                            language: str = "fr", stream: bool = True,
                            callbacks: Optional[SpeechCallbacks] = None) -> AsyncGenerator[StreamResponse, None]:
        try:
            await self.init_tts(model_type, voice, language)
            model_config = self.tts_config[model_type]
            
            if callbacks and callbacks.on_progress:
                callbacks.on_progress(0, "Démarrage de la synthèse...")
            yield StreamResponse(type="progress", content="0", metadata={"message": "Démarrage..."})

            if self._stop_event.is_set() or (callbacks and callbacks.should_stop()):
                yield StreamResponse(type="status", content="stopped")
                return

            total_steps = len(text.split())
            for step in range(total_steps):
                progress = ((step + 1) / total_steps) * 100
                if stream:
                    yield StreamResponse(type="progress", content=progress, metadata={"message": "Synthèse en cours..."})
                    await asyncio.sleep(0.1)
            
            # Génération audio selon le type de modèle
            if model_config["type"] == "multi_speaker":
                base_voice_path = path.join(path.dirname(current_file), "voices")
                if voice:
                    voice_path = path.join(base_voice_path, voice+".wav")
                wav = self.tts_model.tts(text=text, speaker_wav=voice_path, language=language)
            else:
                wav = self.tts_model.tts(text=text)

            # Conversion du numpy array en bytes
            wav_bytes = io.BytesIO()
            sf.write(wav_bytes, wav, self.tts_model.synthesizer.output_sample_rate, format='WAV')
            wav_bytes = wav_bytes.getvalue()

            if callbacks and callbacks.on_complete:
                callbacks.on_complete(wav_bytes)

            yield StreamResponse(type="audio", content=wav_bytes)
            yield StreamResponse(type="status", content="completed", metadata={"audio": wav_bytes})

        except Exception as e:
            if callbacks and callbacks.on_error:
                callbacks.on_error(e)
            yield StreamResponse(type="error", content=str(e))
        finally:
            self.reset()

# Instance globale
speech_processor = SpeechProcessor()