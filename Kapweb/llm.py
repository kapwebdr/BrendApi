from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from os import path
import json
from Kapweb.session import UserSession

current_file = path.realpath(__file__)

def load_models(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data

brenda_llm_config = {"max_new_tokens":2048,"context_length":2048,"temperature":0.7,"stream":True}
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
    
    # if history:
    #     final_prompt += "Contexte précédent:\n" + "\n".join(history) + "\n\n'
    
    final_prompt += f"{current_prompt}"
    
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
        n_threads=6,      # Limiter le nombre de threads
        use_mlock=True,   # Améliorer la gestion mémoire
        use_mmap=True,    # Utiliser mmap pour le chargement
        seed=42           # Fixer une seed pour la reproductibilité
    )
    
    return llm

async def generate_stream(prompt, session: UserSession, model_name=None, models=None):
    try:
        # Vérifier si le modèle doit être chargé
        if (not session.llm or session.current_model != model_name) and model_name in models:
            llm = loadLlm(models[model_name])
            if not llm:
                yield f'data: {{"error": "Échec du chargement du modèle"}}\n\n'
                return
            
            session.llm = llm
            session.current_model = model_name
            session.loaded_model_config = models[model_name]

        for chunk in session.llm.stream(prompt):
            if chunk:
                yield f'data: {chunk}\n\n'
        
        yield 'data: [DONE]\n\n'
    except Exception as e:
        try:
            session.llm.stop()
        except:
            pass
        raise e