# Brenda AI API

Brenda AI API est une interface de programmation unifiée qui combine plusieurs outils d'intelligence artificielle, de traitement multimédia et d'analyse système.

## 🎯 À propos

Développé par Damien RICHARD ([Kapweb](http://www.kapweb.com)), ce projet vise à fournir une API REST unifiée pour accéder à différents services d'IA et de traitement de données.

## 🌟 Fonctionnalités

- **LLM (Large Language Models)**
  - Support de multiples modèles (Vigogne, CodeLlama, etc.)
  - Streaming des réponses
  - Gestion de session
  - Système de prompt personnalisable

- **Traitement d'Images**
  - Génération d'images avec SDXL
  - Raffinement d'images existantes
  - Analyse d'images avec CLIP
  - OCR (Reconnaissance de texte)

- **Audio**
  - Synthèse vocale (TTS) avec XTTS-v2 et MeloTTS
  - Reconnaissance vocale avec Faster-Whisper
  - Support multilingue

- **HTTP et Média**
  - Streaming de contenu web
  - Extraction de contenu HTML
  - Streaming YouTube
  - Analyse d'URL

- **Système**
  - Monitoring CPU/RAM/GPU
  - Gestion des ressources
  - Métriques en temps réel

## 🚀 Installation

### Prérequis

- Python 3.11+
- CUDA compatible GPU (optionnel)
- Apple Silicon (MPS) ou NVIDIA GPU recommandé

### Installation des dépendances
bash
pip install -r requirements.txt


## 📦 Dépendances Principales

- **LLM et IA**
  - llama-cpp-python
  - langchain
  - langchain_community
  - transformers
  - accelerate
  - torch

- **Image**
  - Pillow
  - diffusers
  - pytesseract

- **Audio**
  - TTS
  - faster-whisper
  - melo-tts

- **Web et API**
  - fastapi
  - uvicorn[standard]
  - aiohttp
  - websockets
  - beautifulsoup4
  - pytube

- **Système**
  - psutil
  - gputil

## 🎮 Utilisation

### Lancement de l'API
bash
./launch-api.sh
ou
python Brendapi.py

L'API sera accessible sur `http://localhost:8000`

### Endpoint Principal

Toutes les fonctionnalités sont accessibles via l'endpoint unifié :

POST /v1/ai/process

Voir `services.api.txt` pour la documentation complète des endpoints.

## 🔧 Configuration

Les configurations des modèles sont stockées dans :
- `models.json` : Configuration des modèles LLM
- `image_models.json` : Configuration des modèles de génération d'images

## 📝 Licence

Copyright 2024 Damien RICHARD - Kapweb

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## 🙏 Remerciements

Ce projet utilise plusieurs projets open source remarquables :

- [LLaMA](https://github.com/facebookresearch/llama) - Meta AI
- [Langchain](https://github.com/hwchase17/langchain)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Hugging Face](https://huggingface.co/) et leur écosystème
- [XTTS-v2](https://github.com/coqui-ai/TTS)
- [MeloTTS](https://github.com/myshell-ai/MeloTTS)
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper)
- [Stable Diffusion](https://stability.ai/)

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.

## 📞 Contact

- **Auteur** : Damien RICHARD
- **Entreprise** : Kapweb
- **Site Web** : [www.kapweb.com](http://www.kapweb.com)

## 📚 Documentation

Pour une documentation détaillée des endpoints et des fonctionnalités, consultez :
- `api.txt` : Documentation complète de l'API
- Les commentaires dans le code source

## 🔍 Exemples d'utilisation

Voir le fichier `api.txt` pour des exemples détaillés d'utilisation de chaque endpoint.