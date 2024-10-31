import aiohttp
import os
from os import path

async def download_model(model):
    base_url = "https://huggingface.co"
    model_url = f"{base_url}/{model['model_name']}/resolve/main/{model['model_file']}"
    save_path = path.join(path.dirname(__file__), "..", "Cache", "LlamaCppModel", 
                         model['model_name'].replace('/', path.sep), model['model_file'])
    temp_path = save_path + '.downloading'

    print(f"\n=== Téléchargement du modèle {model['model_name']} ===")
    print(f"URL: {model_url}")
    print(f"Chemin de sauvegarde: {save_path}")
    print(f"Fichier temporaire: {temp_path}")

    async def get_remote_size():
        print("\nVérification de la taille du fichier distant...")
        async with aiohttp.ClientSession() as session:
            headers = {'Range': 'bytes=0-0'}
            async with session.get(model_url, headers=headers) as response:
                print(f"Code de statut: {response.status}")
                print(f"Headers reçus: {dict(response.headers)}")
                
                if response.status not in [200, 206]:
                    raise Exception(f"Erreur HTTP {response.status}")
                
                content_range = response.headers.get('content-range', '')
                if content_range:
                    total_size = int(content_range.split('/')[-1])
                    print(f"Taille totale (depuis content-range): {total_size} octets")
                    return total_size
                else:
                    size = int(response.headers.get('content-length', 0))
                    print(f"Taille totale (depuis content-length): {size} octets")
                    return size

    try:
        remote_size = await get_remote_size()
        
        if path.exists(save_path):
            local_size = path.getsize(save_path)
            print(f"\nFichier existant trouvé:")
            print(f"Taille locale: {local_size} octets")
            print(f"Taille distante: {remote_size} octets")
            
            if local_size == remote_size:
                print("Le fichier est déjà complet!")
                yield f"data: {{'status': 'exists', 'path': '{save_path}'}}\n\n"
                return
            else:
                print("Le fichier existant est incomplet ou corrompu")
            
        start_byte = 0
        if path.exists(temp_path):
            start_byte = path.getsize(temp_path)
            print(f"\nFichier temporaire trouvé: {start_byte} octets déjà téléchargés")
            
            if start_byte > remote_size:
                print("Fichier temporaire corrompu, suppression...")
                os.remove(temp_path)
                start_byte = 0
        
        os.makedirs(path.dirname(save_path), exist_ok=True)
        
        headers = {}
        if start_byte > 0:
            headers['Range'] = f'bytes={start_byte}-'
            print(f"\nReprise du téléchargement à partir de {start_byte} octets")
            yield f"data: {{'status': 'resuming', 'progress': {start_byte/remote_size:.2f}}}\n\n"
        else:
            print("\nDémarrage d'un nouveau téléchargement")

        async with aiohttp.ClientSession() as session:
            async with session.get(model_url, headers=headers) as response:
                print(f"Code de statut pour le téléchargement: {response.status}")
                
                if response.status not in [200, 206]:
                    raise Exception(f"Erreur HTTP {response.status}")
                
                mode = 'ab' if start_byte > 0 else 'wb'
                print(f"Mode d'écriture: {mode}")
                
                with open(temp_path, mode) as file:
                    download_size = start_byte
                    chunk_count = 0
                    async for chunk in response.content.iter_chunked(8192):
                        file.write(chunk)
                        download_size += len(chunk)
                        chunk_count += 1
                        if chunk_count % 1000 == 0:
                            print(f"Progression: {download_size}/{remote_size} octets ({(download_size/remote_size*100):.1f}%)")
                        progress = download_size / remote_size
                        yield f"data: {{'progress': {progress:.2f}}}\n\n"
                
                print(f"\nTéléchargement terminé: {download_size}/{remote_size} octets")
                
                if download_size != remote_size:
                    raise Exception(f"Téléchargement incomplet: {download_size}/{remote_size} octets")
                
                print("Renommage du fichier temporaire en fichier final...")
                os.rename(temp_path, save_path)
                print("Téléchargement complété avec succès!")
                yield f"data: {{'status': 'completed', 'path': '{save_path}'}}\n\n"

    except Exception as e:
        print(f"\nERREUR: {str(e)}")
        if path.exists(save_path) and path.getsize(save_path) != remote_size:
            print(f"Suppression du fichier incomplet: {save_path}")
            os.remove(save_path)
        print("Conservation du fichier temporaire pour une reprise ultérieure")
        yield f"data: {{'error': 'Erreur lors du téléchargement du fichier : {str(e)}'}}\n\n" 