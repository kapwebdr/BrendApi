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

def create_chain(llm, system_message=None):
    return llm

def format_prompt(messages, system_message, prompt_template=None):
    print("\n=== Formatage du Prompt ===")
    print(f"Message système reçu: {system_message}")
    
    # Construction de l'historique et du prompt
    history = []
    current_prompt = ""
    
    print("\nTraitement des messages:")
    for message in messages:
        print(f"\nMessage: {message}")
        if message['role'] in ['user', 'human']:
            current_prompt = message['content']
            print(f"Message utilisateur: {current_prompt}")
        elif message['role'] in ['assistant', 'ai']:
            history.append(message['content'])
            print(f"Ajout à l'historique: {message['content']}")
    
    # Assemblage du prompt final
    final_prompt = f"{system_message}\n\n"
    
    if history:
        final_prompt += "Contexte précédent:\n" + "\n".join(history) + "\n\n"
    
    final_prompt += f"Question: {current_prompt}"
    
    print("\nPrompt final:")
    print(final_prompt)
    
    return final_prompt

def loadLlm(model):
    model_path = path.join(path.dirname(current_file), "..", "Cache", "LlamaCppModel", 
                          model['model_name'].replace('/', path.sep), model['model_file'])
    if not path.exists(model_path):
        return None, None

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
        verbose=True,  # Ajout pour le debug
        n_threads=6,   # Limite le nombre de threads
        use_mlock=True,  # Améliore la gestion mémoire
        use_mmap=True    # Utilise mmap pour le chargement du modèle
    )
    
    return llm, llm

async def generate_stream(prompt, session: UserSession, model_name=None, models=None):
    try:
        # Vérifier si le modèle doit être chargé
        if (not session.llm_instance or session.current_model != model_name) and model_name in models:
            print(f"\nChargement automatique du modèle {model_name}")
            chain, llm = loadLlm(models[model_name])
            if not chain or not llm:
                yield f"data: {{'error': 'Échec du chargement du modèle'}}\n\n"
                return
            
            session.llm_instance = chain
            session.llm = llm
            session.current_model = model_name
            session.loaded_model_config = models[model_name]
            print(f"Modèle {model_name} chargé avec succès")

        config = {"configurable": {"session_id": session.session_id}}
        print(session.llm_instance)
        print(prompt)
        print(config)
        for chunk in session.llm_instance.stream(prompt):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        try:
            session.llm_instance.stop()
        except:
            pass
        raise e