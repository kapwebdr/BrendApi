import torch
from faster_whisper import WhisperModel
from TTS.api import TTS
import io
import base64
import soundfile as sf
from os import path
import asyncio
import os

current_file = path.realpath(__file__)

class SpeechProcessor:
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or path.join(path.dirname(current_file), "..", "Cache")
        self.device = "cpu" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float32 if self.device == "mps" else torch.float16
        self.tts_model = None
        self.stt_model = None
        self.current_tts_model = None
        self.current_stt_model = None

    async def init_tts(self, model_name="tts_models/multilingual/multi-dataset/xtts_v2"):
        """Initialise le modèle TTS"""
        try:
            if self.current_tts_model == model_name and self.tts_model:
                return

            print(f"\nChargement du modèle TTS: {model_name}")
            self.tts_model = TTS(model_name)
            self.tts_model.to(self.device)
            self.current_tts_model = model_name
            print(f"Modèle TTS chargé avec succès sur {self.device}")

        except Exception as e:
            print(f"Erreur lors du chargement du modèle TTS: {str(e)}")
            raise

    async def init_stt(self, model_size="large-v3"):
        """Initialise le modèle STT (Whisper)"""
        try:
            if self.current_stt_model == model_size and self.stt_model:
                return

            print(f"\nChargement du modèle STT: {model_size}")
            compute_type = "float32" if self.device == "mps" else "float16"
            self.stt_model = WhisperModel(
                model_size,
                device=self.device,
                compute_type=compute_type,
                download_root=path.join(self.cache_dir, "WhisperCache")
            )
            self.current_stt_model = model_size
            print(f"Modèle STT chargé avec succès sur {self.device}")

        except Exception as e:
            print(f"Erreur lors du chargement du modèle STT: {str(e)}")
            raise

    async def text_to_speech(self, text, voice_path=None, language="fr"):
        """Convertit du texte en audio avec streaming"""
        # Définir le chemin de base pour les voix
        base_voice_path = path.join(path.dirname(current_file), "voices")
        if voice_path:
            voice_path = path.join(base_voice_path, voice_path)
        print(voice_path)    
        try:
            if not self.tts_model:
                raise ValueError("Le modèle TTS n'est pas chargé")

            # Générer l'audio en mémoire
            audio_buffer = io.BytesIO()
            
            # Progression de la génération
            total_steps = len(text.split()) # Estimation basée sur le nombre de mots
            for step in range(total_steps):
                progress = ((step + 1) / total_steps) * 100
                yield f'data: {{"progress": {progress:.2f}}}\n\n'
                await asyncio.sleep(0.1)

            # Générer l'audio
            if voice_path:
                wav = self.tts_model.tts(
                    text=text,
                    speaker_wav=voice_path,
                    language=language
                )
            else:
                wav = self.tts_model.tts(text=text, language=language)

            # Sauvegarder dans le buffer
            sf.write(audio_buffer, wav, self.tts_model.synthesizer.output_sample_rate, format='WAV')
            audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode()

            yield f'data: {{"status": "completed", "audio": "{audio_base64}", "sample_rate": {self.tts_model.synthesizer.output_sample_rate}}}\n\n'

        except Exception as e:
            yield f'data: {{"error": "Erreur lors de la synthèse vocale : {str(e)}"}}\n\n'

    async def speech_to_text(self, audio_data):
        """Convertit l'audio en texte"""
        try:
            if not self.stt_model:
                raise ValueError("Le modèle STT n'est pas chargé")

            # Sauvegarder l'audio temporairement en mémoire
            audio_buffer = io.BytesIO(audio_data)
            audio_array, sample_rate = sf.read(audio_buffer)
            
            # Sauvegarder temporairement pour Whisper
            temp_buffer = io.BytesIO()
            sf.write(temp_buffer, audio_array, sample_rate, format='WAV')
            temp_buffer.seek(0)

            # Transcription
            segments, info = self.stt_model.transcribe(
                temp_buffer,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            return {
                "text": " ".join([segment.text for segment in segments]),
                "language": info.language,
                "segments": [
                    {
                        "text": segment.text,
                        "start": segment.start,
                        "end": segment.end
                    }
                    for segment in segments
                ]
            }

        except Exception as e:
            return {"error": f"Erreur lors de la transcription : {str(e)}"}

    def get_available_models(self):
        """Retourne la liste des modèles disponibles"""
        return {
            "tts": {
                "xtts_v2": {
                    "name": "XTTS v2",
                    "languages": ["fr", "en", "es", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn"],
                    "voices":
                        [
                            {
                                "path": "morgan.wav",
                                "label":"Morgan Freeman"
                            }
                        ]
                }
            },
            "stt": {
                "whisper": {
                    "name": "Faster Whisper",
                    "sizes": ["tiny", "base", "small", "medium", "large-v3"]
                }
            }
        }

# Instance globale du processeur audio
speech_processor = SpeechProcessor() 