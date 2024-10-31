import torch
import soundfile as sf
from transformers import pipeline
from faster_whisper import WhisperModel
from datasets import load_dataset
from TTS.api import TTS
from os import path

current_file = path.realpath(__file__)
#https://tts.readthedocs.io/en/latest/index.html
#https://huggingface.co/facebook/musicgen-stereo-large
#https://github.com/SYSTRAN/faster-whisper

def Stt(source,device="mps",model_size="large-v3",compute_type="float32"):
    model = WhisperModel(model_size, device=device,compute_type=compute_type, download_root=path.join(path.dirname(current_file),"..", "Cache")) # compute_type="float16",
    segments, info = model.transcribe(path.join(path.dirname(current_file),"..", source), beam_size=5, vad_filter=True,vad_parameters=dict(min_silence_duration_ms=500))
    language    = info.language
    # for segment in segments:
    #     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    return segments,language

def pipetts(text,modelname="microsoft/speecht5_tts", dataset='Matthijs/cmu-arctic-xvectors',output="output"):
    synthesiser             = pipeline("text-to-speech",modelname)
    embeddings_dataset      = load_dataset(dataset, split="validation")
    speaker_embedding       = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    speech                  = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})
    sf.write(path.join(path.dirname(current_file),"..", "Outputs", output + ".wav"), speech["audio"], samplerate=speech["sampling_rate"])

def Xtts(text,modelname,voice="morgan",output="output",output_dir="Outputs",language="fr", device='cpu'):
    tts = TTS(modelname)
    tts.to(device)
    tts.tts_to_file(text=text,
                    file_path=path.join(path.dirname(current_file),"..", output_dir, output + ".wav"),
                    speaker_wav=path.join(path.dirname(current_file), "voices", voice + ".wav"),
                    language=language)
    return output + ".wav"
    
def Tts(text,modelname,output="output",voice_dir="voices",speaker="ljspeech", device='cpu'):
    print(path.join(path.dirname(current_file), "voices", voice_dir, speaker))
    tts = TTS(modelname)
    tts.to(device)
    tts.tts_to_file(text=text,
                    file_path=path.join(path.dirname(current_file),"..", "Outputs", output + ".wav"),
                    voice_dir=path.join(path.dirname(current_file), "voices", voice_dir),
                    speaker=path.join(path.dirname(current_file), "voices", voice_dir,speaker))
    
def Bark(text,output="output",speaker="ljspeech", device='cpu'):
    Tts(text,"tts_models/multilingual/multi-dataset/bark",output,"bark/",speaker, device)

def Music(prompt,modelname= "facebook/musicgen-stereo-small",output="output", device='cpu'):
    synthesiser = pipeline("text-to-audio",modelname, device=device, torch_dtype=torch.float32)
    music = synthesiser(prompt, forward_params={"max_new_tokens": 256})
    file = path.join(path.dirname(current_file),"..", "Outputs",output+".wav")
    sf.write(file, music["audio"][0].T, music["sampling_rate"])
    return file