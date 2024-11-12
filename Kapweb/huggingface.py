import aiohttp
import os
from os import path

async def get_model_files(model_name):
    async with aiohttp.ClientSession() as session:
        api_url = f"https://huggingface.co/api/models/{model_name}"
        async with session.get(api_url) as response:
            if response.status == 200:
                data = await response.json()
                files = [file['rfilename'] for file in data.get("siblings", [])]
                return files
            else:
                raise Exception(f"Erreur lors de la récupération des fichiers du modèle: {response.status}")
            
async def download_model(model):
    base_url = "https://huggingface.co"
    model_name = model['model_name']
    print(model['model_file'])
    # Déterminer les fichiers à télécharger
    if 'model_file' in model and isinstance(model['model_file'], list):
        all_files = await get_model_files(model_name)
        print(all_files,model['model_file'])
        model_files = [f for f in all_files if f in model['model_file']]
        missing_files = [f for f in model['model_file'] if f not in model_files]
        if missing_files:
            print(f"Fichiers manquants dans le modèle Hugging Face : {missing_files}")
        
    elif 'model_file' in model and isinstance(model['model_file'], str):
        # Si model_file est une chaîne, télécharger uniquement ce fichier
        model_files = [model['model_file']]
    else:
        # Sinon, télécharger tous les fichiers du modèle
        model_files = await get_model_files(model_name)
    print(model_files)

    async def get_remote_size(model_url):
        async with aiohttp.ClientSession() as session:
            headers = {'Range': 'bytes=0-0'}
            async with session.get(model_url, headers=headers) as response:
                if response.status not in [200, 206]:
                    raise Exception(f"Erreur HTTP {response.status}")
                
                content_range = response.headers.get('content-range', '')
                return int(content_range.split('/')[-1]) if content_range else int(response.headers.get('content-length', 0))

    # Télécharger chaque fichier
    for model_file in model_files:
        model_url = f"{base_url}/{model_name}/resolve/main/{model_file}"
        save_path = path.join(path.dirname(__file__), "..", "Cache", "LlamaCppModel", 
                              model_name.replace('/', path.sep), model_file)
        temp_path = save_path + '.downloading'
        
        print(f'Downloading {model_url}')
        
        try:
            remote_size = await get_remote_size(model_url)
            
            if path.exists(save_path) and path.getsize(save_path) == remote_size:
                yield f'data: {{"status": "exists", "path": "{save_path}"}}\n\n'
                continue
            
            start_byte = 0
            if path.exists(temp_path):
                start_byte = path.getsize(temp_path)
                
                if start_byte > remote_size:
                    os.remove(temp_path)
                    start_byte = 0
            
            os.makedirs(path.dirname(save_path), exist_ok=True)
            headers = {'Range': f'bytes={start_byte}-'} if start_byte > 0 else {}
            if start_byte > 0:
                yield f'data: {{"status": "resuming", "progress": {start_byte/remote_size:.2f}}}\n\n'
            
            async with aiohttp.ClientSession() as session:
                async with session.get(model_url, headers=headers) as response:
                    if response.status not in [200, 206]:
                        raise Exception(f"Erreur HTTP {response.status}")
                    
                    mode = 'ab' if start_byte > 0 else 'wb'
                    with open(temp_path, mode) as file:
                        download_size = start_byte
                        progress = 0
                        async for chunk in response.content.iter_chunked(8192):
                            file.write(chunk)
                            download_size += len(chunk)
                            new_progress = (download_size / remote_size * 100)
                            if new_progress - progress >= 1:
                                progress = new_progress
                                yield f'data: {{"progress": {progress:.2f}}}\n\n'
                    
                    if download_size != remote_size:
                        raise Exception(f"Téléchargement incomplet: {download_size}/{remote_size} octets")
                    
                    os.rename(temp_path, save_path)
                    yield f'data: {{"status": "downloaded", "path": "{save_path}"}}\n\n'
        
        except Exception as e:
            if path.exists(save_path) and path.getsize(save_path) != remote_size:
                os.remove(save_path)
            yield f'data: {{"error": "Erreur lors du téléchargement du fichier : {str(e)}"}}\n\n'