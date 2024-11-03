import aiohttp
import os
from os import path

async def download_model(model):
    base_url = "https://huggingface.co"
    model_url = f"{base_url}/{model['model_name']}/resolve/main/{model['model_file']}"
    save_path = path.join(path.dirname(__file__), "..", "Cache", "LlamaCppModel", 
                         model['model_name'].replace('/', path.sep), model['model_file'])
    temp_path = save_path + '.downloading'

    async def get_remote_size():
        async with aiohttp.ClientSession() as session:
            headers = {'Range': 'bytes=0-0'}
            async with session.get(model_url, headers=headers) as response:
                
                if response.status not in [200, 206]:
                    raise Exception(f"Erreur HTTP {response.status}")
                
                content_range = response.headers.get('content-range', '')
                if content_range:
                    total_size = int(content_range.split('/')[-1])
                    return total_size
                else:
                    size = int(response.headers.get('content-length', 0))
                    return size

    try:
        remote_size = await get_remote_size()
        
        if path.exists(save_path):
            local_size = path.getsize(save_path)
            
            if local_size == remote_size:
                yield f'data: {{"status": "exists", "path": "{save_path}"}}\n\n'
                return
            
        start_byte = 0
        if path.exists(temp_path):
            start_byte = path.getsize(temp_path)
            
            if start_byte > remote_size:
                os.remove(temp_path)
                start_byte = 0
        
        os.makedirs(path.dirname(save_path), exist_ok=True)
        
        headers = {}
        if start_byte > 0:
            headers['Range'] = f'bytes={start_byte}-'
            yield f'data: {{"status": "resuming", "progress": {start_byte/remote_size:.2f}}}\n\n'
       
        async with aiohttp.ClientSession() as session:
            async with session.get(model_url, headers=headers) as response:
                
                if response.status not in [200, 206]:
                    raise Exception(f"Erreur HTTP {response.status}")
                
                mode = 'ab' if start_byte > 0 else 'wb'
                
                with open(temp_path, mode) as file:
                    download_size = start_byte
                    chunk_count = 0
                    progress = 0
                    async for chunk in response.content.iter_chunked(8192):
                        file.write(chunk)
                        download_size += len(chunk)
                        chunk_count += 1
                        new_progress = (download_size/remote_size*100)
                        if new_progress - progress >= 1:
                            progress = new_progress
                            yield f'data: {{"progress": {progress:.2f}}}\n\n'
                
                if download_size != remote_size:
                    raise Exception(f"Téléchargement incomplet: {download_size}/{remote_size} octets")
                
                os.rename(temp_path, save_path)
                yield f'data: {{"status": "completed", "path": "{save_path}"}}\n\n'

    except Exception as e:
        if path.exists(save_path) and path.getsize(save_path) != remote_size:
            os.remove(save_path)
        yield f'data: {{"error": "Erreur lors du téléchargement du fichier : {str(e)}"}}\n\n'