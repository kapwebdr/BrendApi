from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from os import path
import json
from Kapweb.session import UserSession
import os
import re

current_file = path.realpath(__file__)

# Définition des motifs de ponctuation pour le découpage
SENTENCE_ENDINGS = r'[.!?]'
PHRASE_BREAKS = r'[,;:]'
PUNCTUATION_PATTERN = f"({SENTENCE_ENDINGS}|{PHRASE_BREAKS})"

def format_chunk(chunk: str) -> str:
    """Formate un chunk de texte pour le streaming en échappant les caractères spéciaux"""
    return chunk.replace('\n', '\\n')\
                .replace('\r', '\\r')\
                .replace('\t', '\\t')\
                .replace('"', '\\"')\

def load_models(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data

brenda_llm_config = {"max_new_tokens":2048,"context_length":2048,"temperature":0.5,"stream":True}
brenda_system = "Tu es Brenda, mon assistante, secrétaire personnelle."
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

def format_prompt(messages, system_message, prompt_template=None):
    # Construction de l'historique et du prompt
    history = []
    current_prompt = ""
    
    for message in messages:
        if message['role'] in ['user', 'human']:
            current_prompt = message['content']
        elif message['role'] in ['assistant', 'ai']:
            history.append(message['content'])
    
    # Assemblage du prompt final
    final_prompt = f"{system_message}\n\n"
    print(final_prompt)
    return final_prompt

def loadLlm(model):
    model_path = path.join(path.dirname(current_file), "..", "Cache", "LlamaCppModel", 
                          model['model_name'].replace('/', path.sep), model['model_file'])
    if not path.exists(model_path):
        return None

    n_gpu_layers = 1
    n_batch = 4096
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=n_batch,
        f16_kv=True,
        config=brenda_llm_config,
        callback_manager=callback_manager,
        streaming=True,
        verbose=True,
    )
    
    return llm

async def generate_stream(prompt, session: UserSession, model_name=None, models=None, format_type="chunk"):
    """
    Génère un stream de texte avec différentes options de formatage.
    
    Args:
        prompt: Le prompt à traiter
        session: La session utilisateur
        model_name: Le nom du modèle à utiliser
        models: La configuration des modèles
        format_type: Le type de formatage ("chunk", "line", "encoded", "speech")
            - "chunk": Envoie les chunks bruts
            - "line": Envoie ligne par ligne
            - "encoded": Envoie les chunks avec les retours à la ligne encodés (\n)
            - "speech": Découpe intelligent pour la synthèse vocale
    """
    print(f"Format type: {format_type}")
    try:
        llm = loadLlm(models[model_name])
        print(f"Prompt: {prompt}")
        
        if format_type == "speech":
            buffer = ""
            for chunk in llm.stream(prompt):
                if chunk:
                    buffer += chunk
                    matches = list(re.finditer(PUNCTUATION_PATTERN, buffer))
                    if matches:
                        last_match = matches[-1]
                        split_point = last_match.end()
                        
                        sentence = buffer[:split_point].strip()
                        buffer = buffer[split_point:].strip()
                        
                        if sentence:
                            pause_type = "long" if re.search(SENTENCE_ENDINGS, sentence[-1]) else "short"
                            yield f'data: {{"text": "{format_chunk(sentence)}", "pause": "{pause_type}"}}\n\n'
            
            if buffer.strip():
                yield f'data: {{"text": "{format_chunk(buffer.strip())}", "pause": "none"}}\n\n'
                
        elif format_type == "line":
            buffer = ""
            for chunk in llm.stream(prompt):
                if chunk:
                    buffer += chunk
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if line.strip():
                            yield f'data: {{"text": "{format_chunk(line)}"}}\n\n'
                    
            if buffer.strip():
                yield f'data: {{"text": "{format_chunk(buffer)}"}}\n\n'
                
        elif format_type == "encoded":
            for chunk in llm.stream(prompt):
                if chunk:
                    yield f'data: {{"text": "{format_chunk(chunk)}"}}\n\n'
                    
        else:  # format_type == "chunk" (défaut)
            for chunk in llm.stream(prompt):
                if chunk:
                    yield f'data: {{"text": "{format_chunk(chunk)}"}}\n\n'
        
        yield 'data: [DONE]\n\n'
    except Exception as e:
        print(f"Erreur pendant le streaming: {str(e)}")
        try:
            session.llm.stop()
        except:
            pass
        raise e