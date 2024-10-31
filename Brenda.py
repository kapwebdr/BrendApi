import streamlit as st

import json 
from os import path
import requests
import os
import pickle
from Kapweb.dispatcher import Dispatch, DispatchSpacy
from Kapweb.agent import Agent
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from datetime import datetime
import pandas as pd

current_file = path.realpath(__file__)

def load_models(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data

models              = load_models(path.join(path.dirname(current_file), "models.json"))
brenda_model        = None
brenda_title        = '💬 Brenda Kapweb ChatBot'
brenda_logo         = 'brenda.png'
brenda_welcome      = "Hé c'est Brenda, pose tes questions !"
brenda_system       = "Tu es Brenda, mon assistante, secrétaire personnelle."

brenda_agent        = Agent()
brenda_llm_config   = {"max_new_tokens":2048,"context_length":2048,"temperature":0.5,"stream":True}

st.set_page_config(page_title=brenda_title)

waiting             = st.empty()  
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
def listHistoryFiles():
    directory_name = os.path.join(os.path.dirname(current_file),"History")
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)
    file_list = []
    for root, dirs, files in os.walk(directory_name):
        for file in files:
            if file.endswith(".pkl"):
                file_list.append(file.replace(".pkl",''))
    sorted_file_list = sorted(file_list, key=lambda date: datetime.strptime(date, "%Y-%m-%d-%H:%M:%S"), reverse=True)
    return sorted_file_list

def loadHistory(name):
    st.session_state.pickle_file =  os.path.join(os.path.dirname(current_file),"History", name+".pkl")
    if os.path.exists(st.session_state.pickle_file):
        with open(st.session_state.pickle_file, "rb") as f:
            return pickle.load(f)

def newHistory():
    now = datetime.now()
    return os.path.join(os.path.dirname(current_file),"History", now.strftime("%Y-%m-%d-%H:%M:%S")+".pkl")

def deleteHistory(name):
    return os.remove(os.path.join(os.path.dirname(current_file),"History", name+".pkl"))

def selectBoxHistory(container,options,index=None):
    #container.empty()
    with container:
        return st.selectbox(
            "Historique de conversation",
            options,
            index=index,
            placeholder="Choix de l'Historique de conversation",
            key="history_file"
            )

if "pickle_file" not in st.session_state.keys():
    st.session_state.pickle_file = newHistory()

history_files   = listHistoryFiles()

def download_model(model,progress_bar):
    base_url = "https://huggingface.co"
    model_url = f"{base_url}/{model['model_name']}/resolve/main/{model['model_file']}"  # Vérifiez si cette URL est correcte pour le téléchargement direct
    save_path = os.path.join(os.path.dirname(current_file), "Cache","LlamaCppModel", model['model_name'].replace('/', os.sep), model['model_file'])

    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            response = requests.get(model_url, stream=True)
            total_length = int(response.headers.get('content-length'))

            if total_length is None:  # pas de contenu length header
                progress_bar.empty()
            else:
                download_size = 0
                with open(save_path, 'wb') as file:
                    for data in response.iter_content(chunk_size=4096):
                        download_size += len(data)
                        file.write(data)
                        progress = download_size / total_length
                        progress_bar.progress(progress,text="Téléchargement du modèle.")
        except requests.RequestException as e:
            print(f"Erreur lors du téléchargement du fichier : {e}")
            return None
    return save_path

def loadLlm(model):
    with waiting.container():
        progress_bar = st.progress(0,text="Téléchargement du modèle.")
        model_path = download_model(model,progress_bar)

        # Vérifier si le téléchargement a réussi
        if model_path is None:
            st.error("Erreur lors du téléchargement du modèle.")
            return None

        # Configuration du modèle LLM
        n_gpu_layers = 1  # Metal set to 1 is enough.
        n_batch = 4096  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
        # Chargement du modèle LLM
        llm = LlamaCpp(model_path=model_path,
                        n_gpu_layers=n_gpu_layers,
                        n_batch=n_batch,
                        n_ctx=n_batch,
                        f16_kv=True,
                        config=brenda_llm_config,
                        callback_manager=callback_manager,
                        verbose=True,
                        streaming=True)
        progress_bar.empty()
    waiting.empty()
    return llm

def dynamicStMedia(content):
    if content.startswith("markdown:"):
        st.markdown(content.replace("markdown:", "").strip(),unsafe_allow_html=True)
    elif content.startswith("image:"):
        st.image(content.replace("image:", "").strip())
    elif content.startswith("audio:"):
        st.image(content.replace("audio:", "").strip())
    elif content.startswith("video:"):
        st.image(content.replace("video:", "").strip())
    else:
        st.write(content)

if "history_files" not in st.session_state.keys():
    st.session_state.history_files = listHistoryFiles()

col1, col2 = st.columns(2)
sidebar = st.sidebar
history_file = None
with sidebar:
    st.markdown('<div style="text-align:center"><img src="./app/static/'+brenda_logo+'" height="100" ></div>',unsafe_allow_html=True)
    brenda_system = st.text_input('System', brenda_system,key="brenda_system")
    option = st.selectbox(
        "Choix du model LLM",
        models,
        index=None,
        placeholder="Choix du model LLM..."
        )
    container_selectbox  = st.empty()
    history_file = selectBoxHistory(container_selectbox,st.session_state['history_files'])
    
    if history_file != None:
        if st.button('Supprimer l\'historique'):
            deleteHistory(history_file)
            st.session_state['history_files']  = listHistoryFiles()
            st.session_state.pickle_file = newHistory()
            st.session_state.history = []
            st.session_state.messages = [{"role": "assistant", "content": brenda_welcome}]
            #history_file = selectBoxHistory(container_selectbox,st.session_state['history_files'])
            
    if st.button('Nouvelle conversation'):
        history_file = None
        st.session_state.pickle_file = newHistory()
        st.session_state.history = []
        st.session_state.messages = [{"role": "assistant", "content": brenda_welcome}]

if option != None:
    brenda_model = models[option]
    brenda_llm = loadLlm(brenda_model)
            
if history_file != None:
    print("Load History")
    st.session_state['history']     = loadHistory(history_file)
    st.session_state.messages = []
    for item in st.session_state.history:
        st.session_state.messages.append({"role": "user", "content": item['input']})
        st.session_state.messages.append({"role": "assistant", "content": item['output']})

def llm_response(prompt_input, brenda_system):
    template    = brenda_model['prompt_template'].replace("{system}",brenda_system)
    prompt      = PromptTemplate(input_variables=["history", "prompt"], template=template)
    container = st.empty()  
    text = ''
    history = '\n'.join([f"{item['input']}\n{item['output'].strip()}" for item in st.session_state.history])
    prompt_formated =prompt.format(history=history,prompt=prompt_input)
    #print(brenda_llm.invoke(prompt_formated))
    for chunk in brenda_llm.stream(prompt_formated):
        text += chunk
        container.write(text)
    return text

if "history" not in st.session_state.keys() and option != None:
    if os.path.exists(st.session_state.pickle_file):
        with open(st.session_state.pickle_file, "rb") as f:
            st.session_state.history = pickle.load(f)
    else:
        st.session_state.history = []
    
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": brenda_welcome}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        dynamicStMedia(message["content"])

if option != None:
    if prompt := st.chat_input(placeholder="Ta demande"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    response = ""
    with st.chat_message("assistant"):
        with st.spinner('Je réfléchis. . . Sois patient jeune padawan.'):
            result = Dispatch(prompt)  
            if result is not None:
                if hasattr(brenda_agent, result['agent']):
                    methode     = getattr(brenda_agent, result['agent'])
                    response    = methode(result)
                    dynamicStMedia(response)
                else:
                    st.write(f"Aucune méthode nommée {result['agent']}")
            else:
                response = llm_response(prompt,brenda_system)

    st.session_state.history.append({"input":prompt,"output":response})  
    with open(st.session_state.pickle_file, "wb") as f:
        pickle.dump(st.session_state.history, f)
    st.session_state.history_files = listHistoryFiles()     
    #history_file = selectBoxHistory(container_selectbox,st.session_state['history_files'])    
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
    
    