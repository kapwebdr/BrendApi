from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from os import path
import json
from Kapweb.session import UserSession
import os
import re
from typing import Callable, Optional, Dict, Any, AsyncGenerator
from dataclasses import dataclass
import asyncio

@dataclass
class StreamResponse:
    type: str  # "text", "status", "error"
    content: str
    metadata: Optional[Dict[str, Any]] = None


current_file = path.realpath(__file__)
# Définition des motifs de ponctuation pour le découpage
SENTENCE_ENDINGS = r'[.!?]'
PHRASE_BREAKS = r'[,;:]'
PUNCTUATION_PATTERN = f"({SENTENCE_ENDINGS}|{PHRASE_BREAKS})"

class LLMCallbacks:
    def __init__(self, 
                 on_chunk: Optional[Callable[[str, Dict[str, Any]], None]] = None,
                 on_complete: Optional[Callable[[str], None]] = None,
                 on_error: Optional[Callable[[Exception], None]] = None,
                 should_stop: Optional[Callable[[], bool]] = None):
        self.on_chunk = on_chunk
        self.on_complete = on_complete
        self.on_error = on_error
        self.should_stop = should_stop or (lambda: False)

class LLMGenerator:
    def __init__(self):
        self._stop_event = asyncio.Event()
        self.current_generation = None

    def stop(self):
        """Arrête la génération en cours"""
        if not self._stop_event.is_set():
            self._stop_event.set()

    def reset(self):
        """Réinitialise le générateur"""
        self._stop_event.clear()
        self.current_generation = None

    async def generate_stream(self, prompt, session: UserSession, model_name=None, 
                            models=None, format_type="chunk", 
                            callbacks: Optional[LLMCallbacks] = None) -> AsyncGenerator[StreamResponse, None]:
        """Version améliorée de generate_stream avec support d'arrêt"""
        try:
            self.reset()
            llm = loadLlm(models[model_name])
            self.current_generation = llm
            print(f"Prompt: {prompt}")
            complete_message = ""
            
            for chunk in llm.stream(prompt):
                # Vérifie si on doit arrêter
                if self._stop_event.is_set() or (callbacks and callbacks.should_stop()):
                    print("Génération arrêtée")
                    yield StreamResponse(
                        type="status",
                        content="stopped",
                        metadata={"complete_text": format_chunk(complete_message.strip())}
                    )
                    return

                if chunk:
                    complete_message += chunk
                    if format_type == "speech":
                        buffer = ""
                        for chunk in llm.stream(prompt):
                            if chunk:
                                complete_message += chunk
                                buffer += chunk
                                matches = list(re.finditer(PUNCTUATION_PATTERN, buffer))
                                if matches:
                                    last_match = matches[-1]
                                    split_point = last_match.end()
                                    
                                    sentence = buffer[:split_point].strip()
                                    buffer = buffer[split_point:].strip()
                                    
                                    if sentence:
                                        pause_type = "long" if re.search(SENTENCE_ENDINGS, sentence[-1]) else "short"
                                        response = StreamResponse(
                                            type="text",
                                            content=format_chunk(sentence),
                                            metadata={"pause": pause_type}
                                        )
                                        if callbacks and callbacks.on_chunk:
                                            callbacks.on_chunk(sentence, {"pause": pause_type})
                                        yield response
            
                        if buffer.strip():
                            response = StreamResponse(
                                type="text",
                                content=format_chunk(buffer.strip()),
                                metadata={"pause": "none"}
                            )
                            if callbacks and callbacks.on_chunk:
                                callbacks.on_chunk(buffer.strip(), {"pause": "none"})
                            yield response
                        
                    elif format_type == "line":
                        buffer = ""
                        for chunk in llm.stream(prompt):
                            if chunk:
                                complete_message += chunk
                                buffer += chunk
                                while '\n' in buffer:
                                    line, buffer = buffer.split('\n', 1)
                                    if line.strip():
                                        response = StreamResponse(
                                            type="text",
                                            content=format_chunk(line)
                                        )
                                        if callbacks and callbacks.on_chunk:
                                            callbacks.on_chunk(line, None)
                                        yield response
                        
                        if buffer.strip():
                            response = StreamResponse(
                                type="text",
                                content=format_chunk(buffer)
                            )
                            if callbacks and callbacks.on_chunk:
                                callbacks.on_chunk(buffer, None)
                            yield response
                        
                    else:  # format_type in ["chunk", "encoded"]
                        response = StreamResponse(
                            type="text",
                            content=format_chunk(chunk)
                        )
                        if callbacks and callbacks.on_chunk:
                            callbacks.on_chunk(chunk, None)
                        yield response
        
            # Génération terminée normalement
            final_response = StreamResponse(
                type="status",
                content="completed",
                metadata={"complete_text": format_chunk(complete_message.strip())}
            )
            if callbacks and callbacks.on_complete:
                callbacks.on_complete(complete_message.strip())
            yield final_response

        except Exception as e:
            error_response = StreamResponse(
                type="error",
                content=str(e)
            )
            if callbacks and callbacks.on_error:
                callbacks.on_error(e)
            yield error_response
        finally:
            self.reset()
            if session.llm:
                try:
                    session.llm.stop()
                except:
                    pass

# Instance globale du générateur
llm_generator = LLMGenerator()

def format_chunk(chunk: str) -> str:
    """Formate un chunk de texte pour le streaming en échappant les caractères spéciaux"""
    return chunk.replace('\n', '\\n')\
                .replace('\r', '\\r')\
                .replace('\t', '\\t')\
                .replace('"', '\\"')

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
    
    llm = LlamaCpp(
        model_path=model_path,
        callback_manager=callback_manager,
        max_new_tokens= 16384,        # Augmenté de 2048 à 8192
        max_tokens=16384,
        context_length= 148951,        # Augmenté de 2048 à 16384
        temperature= 0.7,
        stream= True,
        n_ctx= 148951,                  # Contexte maximal
        n_batch= 148951, 
        n_gpu_layers=1, 
        top_p= 0.9,                   
        repeat_penalty= 1.1,
        f16_kv=True,
        streaming=True,
        use_mlock=True,
    )
    
    return llm