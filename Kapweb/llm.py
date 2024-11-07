from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
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


brenda_system = "Tu es Brenda, mon assistante, secrétaire personnelle."
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

def get_prompt_template(template_name: str,_template_path :str=None) -> any:
    """Récupère le template de prompt correspondant au nom donné"""
    template_path = _template_path or path.join(path.dirname(current_file), "..", "models_template.json")
    template_path = '/app/Config/models_template.json'
    with open(template_path, 'r') as file:
        templates = json.load(file)
    return templates.get(template_name)
    

def format_prompt(messages, system_message, prompt_template=None):
    # Construction de l'historique et du prompt
    prompt = prompt_template['pre_prompt'];
    prompt += prompt_template['system_prompt'].format(system=system_message);

    for message in messages:
       if message['role'] in ['user', 'human']:
            prompt += prompt_template['user_prompt'].format(user=message['content']);
       elif message['role'] in ['assistant', 'ai']:
            prompt += prompt_template['assistant_prompt'].format(assistant=message['content']);

    prompt += prompt_template['assistant_prompt'].format(assistant="");
    return prompt

def loadLlm(model):
    model_path = path.join(path.dirname(current_file), "..", "Cache", "LlamaCppModel", 
                          model['model_name'].replace('/', path.sep), model['model_file'])
    if not path.exists(model_path):
        return None
    
    # n_gpu_layers = 1
    llm = LlamaCpp(
        model_path=model_path,
        callback_manager=callback_manager,
        max_new_tokens= 16384,        # Augmenté de 2048 à 8192
        max_tokens=16384,
        context_length= 16384,        # Augmenté de 2048 à 16384
        temperature= 0.7,
        stream= True,
        n_ctx= 4096,                  # Contexte maximal
        n_batch= 4096, 
        n_gpu_layers=1, 
        top_p= 0.9,                   
        repeat_penalty= 1.1,
        f16_kv=True,
        streaming=True,
        use_mlock=True,
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
    """
    try:
        llm = loadLlm(models[model_name])
        print(f"Prompt: {prompt}")
        complete_message = ""  # Pour stocker le message complet
        
        if format_type == "speech":
            buffer = ""
            for chunk in llm.stream(prompt):
                if chunk:
                    complete_message += chunk  # Accumule le message complet
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
                    complete_message += chunk  # Accumule le message complet
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
                    complete_message += chunk  # Accumule le message complet
                    yield f'data: {{"text": "{format_chunk(chunk)}"}}\n\n'
                    
        else:  # format_type == "chunk" (défaut)
            for chunk in llm.stream(prompt):
                if chunk:
                    complete_message += chunk  # Accumule le message complet
                    yield f'data: {{"text": "{format_chunk(chunk)}"}}\n\n'
        
        # Envoie le status completed avec le message complet
        yield f'data: {{"status": "completed", "text": "{format_chunk(complete_message.strip())}"}}\n\n'
    except Exception as e:
        print(f"Erreur pendant le streaming: {str(e)}")
        try:
            session.llm.stop()
        except:
            pass
        raise e