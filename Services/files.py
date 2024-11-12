from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from Kapweb.files import file_manager, StreamResponse, FileCallbacks
from Kapweb.services import ServiceHelper
from typing import List, Optional
import uuid
from datetime import datetime
import httpx
import base64
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

service = ServiceHelper("files")
storage_url = "http://storage:8000"

@app.post("/v1/files/directory/create")
async def create_directory(request: Request):
    print("create_directory")
    data = await request.json()
    if "path" not in data:
        raise HTTPException(status_code=400, detail="Chemin requis")
    
    result = await file_manager.create_directory(data["path"])
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return JSONResponse(content=result)

@app.post("/v1/files/directory/move")
async def move_directory(request: Request):
    data = await request.json()
    if not all(k in data for k in ["source", "destination"]):
        raise HTTPException(status_code=400, detail="Source et destination requises")
    
    result = await file_manager.move_directory(data["source"], data["destination"])
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return JSONResponse(content=result)

@app.post("/v1/files/upload")
async def upload_files(
    request: Request,
    files: Optional[List[UploadFile]] = File(None),
    path: Optional[str] = Form(None)
):
    # Si c'est un upload multipart classique
    if files and path:
        results = []
        for file in files:
            content = await file.read()
            file_path = f"{path}/{file.filename}"
            result = await file_manager.save_file(file_path, content)
            results.append(result)
        return JSONResponse(content={"uploads": results})
    
    # Si c'est un upload base64
    try:
        data = await request.json()
        if all(k in data for k in ["content", "mime_type", "path"]):
            result = await file_manager.save_file_base64(
                data["path"],
                data["content"],
                data["mime_type"]
            )
            return JSONResponse(content={"uploads": [result]})
    except:
        pass
        
    raise HTTPException(
        status_code=400,
        detail="Format invalide. Utilisez soit multipart/form-data avec files et path, soit JSON avec content, mime_type et path"
    )

@app.post("/v1/files/move")
async def move_item(request: Request):
    data = await request.json()
    if "source" not in data or "destination" not in data:
        raise HTTPException(status_code=400, detail="Source et destination requises")
    
    result = await file_manager.move(data["source"], data["destination"])
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return JSONResponse(content=result)

@app.delete("/v1/files/delete")
async def delete_file(request: Request):
    data = await request.json()
    if "path" not in data:
        raise HTTPException(status_code=400, detail="Chemin requis")
    
    result = await file_manager.delete_file(data["path"])
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return JSONResponse(content=result)

@app.delete("/v1/files/directory/delete")
async def delete_directory(request: Request):
    data = await request.json()
    if "path" not in data:
        raise HTTPException(status_code=400, detail="Chemin requis")
    
    result = await file_manager.delete_directory(data["path"])
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return JSONResponse(content=result)

@app.post("/v1/files/download")
async def download_file(request: Request):
    data = await request.json()
    if "path" not in data:
        raise HTTPException(status_code=400, detail="Chemin requis")
    
    try:
        return FileResponse(
            file_manager._validate_path(data["path"]),
            filename=data["path"].split("/")[-1]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/files/stream")
async def stream_file(request: Request):
    data = await request.json()
    if "path" not in data:
        raise HTTPException(status_code=400, detail="Chemin requis")
        
    async def stream_response():
        async for response in file_manager.stream_file(data["path"]):
            if response.type == "data":
                yield f'data: {{"chunk": "{response.content}", "progress": {response.metadata["progress"]}}}\n\n'
            elif response.type == "status":
                if response.content == "completed":
                    yield f'data: {{"status": "completed", "path": "{response.metadata["path"]}", "size": {response.metadata["size"]}}}\n\n'
                else:
                    yield f'data: {{"status": "{response.content}"}}\n\n'
            elif response.type == "error":
                yield f'data: {{"error": "{response.content}"}}\n\n'
        
        yield 'data: [DONE]\n\n'

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream"
    )

@app.post("/v1/files/compress")
async def compress(request: Request):
    data = await request.json()
    if "path" not in data:
        raise HTTPException(status_code=400, detail="Chemin requis")
    
    async def stream_response():
        async for response in file_manager.compress(
            path=data["path"],
            zip_name=data.get("zip_name"),
            is_directory=data.get("is_directory", False)
        ):
            if response.type == "progress":
                yield f'data: {{"progress": {response.content}, "file": "{response.metadata["file"]}"}}\n\n'
            elif response.type == "status":
                if response.content == "completed":
                    yield f'data: {{"status": "completed", "path": "{response.metadata["path"]}", "size": {response.metadata["size"]}}}\n\n'
                else:
                    yield f'data: {{"status": "{response.content}"}}\n\n'
            elif response.type == "error":
                yield f'data: {{"error": "{response.content}"}}\n\n'
    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream"
    )

@app.post("/v1/files/decompress")
async def decompress_zip(request: Request):
    data = await request.json()
    if "path" not in data:
        raise HTTPException(status_code=400, detail="Chemin requis")
    
    result = await file_manager.decompress_zip(
        data["path"],
        data.get("extract_path")
    )
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return JSONResponse(content=result)

@app.post("/v1/files/list")
async def list_directory(request: Request):
    data = await request.json()
    path = data.get("path", "")
    
    result = await file_manager.list_directory(path)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return JSONResponse(content=result)

@app.post("/ready")
async def ready():
    """Endpoint indiquant que le service est prêt"""
    return await service.check_ready()

@app.post("/v1/files/rename")
async def rename_item(request: Request):
    data = await request.json()
    if "path" not in data or "new_name" not in data:
        raise HTTPException(status_code=400, detail="Chemin et nouveau nom requis")
    
    result = await file_manager.rename(data["path"], data["new_name"])
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return JSONResponse(content=result)

@app.post("/v1/files/preview")
async def preview_file(request: Request):
    data = await request.json()
    if "path" not in data:
        raise HTTPException(status_code=400, detail="Chemin requis")
    
    result = await file_manager.preview(data["path"])
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return JSONResponse(content=result)

async def store_in_index(path: str, content: dict) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{storage_url}/v1/storage/set",
            json={
                "key": path,
                "value": content,
                "collection": "files_index"
            }
        )
        return response.json()

@app.post("/v1/files/index")
async def index_item(request: Request):
    data = await request.json()
    if "path" not in data:
        raise HTTPException(status_code=400, detail="Chemin requis")
    
    path = data["path"]
    metadata = data.get("metadata", {})
    is_directory = data.get("is_directory", False)
    
    try:
        full_path = str(file_manager._validate_path(path))
        
        if is_directory:
            content = {
                "path": full_path,
                "type": "directory",
                "metadata": metadata,
                "created_at": datetime.now().isoformat()
            }
        else:
            # Pour les fichiers, on ajoute le contenu en base64
            with open(full_path, 'rb') as f:
                file_content = f.read()
                content = {
                    "path": full_path,
                    "type": "file",
                    "content": base64.b64encode(file_content).decode(),
                    "size": len(file_content),
                    "metadata": metadata,
                    "created_at": datetime.now().isoformat()
                }
        
        result = await store_in_index(path, content)
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/files/index/{path:path}")
async def get_indexed_item(path: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{storage_url}/v1/storage/get/files_index/{path}")
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail="Élément non trouvé")
        return response.json()

@app.get("/v1/files/index")
async def list_indexed_items(type: Optional[str] = None):
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{storage_url}/v1/storage/list/files_index")
        items = response.json()
        
        if type:
            items = [item for item in items if item.get("type") == type]
            
        return JSONResponse(content={"items": items})

@app.delete("/v1/files/index/{path:path}")
async def delete_indexed_item(path: str):
    async with httpx.AsyncClient() as client:
        response = await client.delete(f"{storage_url}/v1/storage/delete/files_index/{path}")
        return response.json()

@app.post("/v1/files/index/search")
async def search_indexed_items(request: Request):
    data = await request.json()
    if "query" not in data:
        raise HTTPException(status_code=400, detail="Query requise")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{storage_url}/v1/storage/search",
            json={
                "query": data["query"],
                "collection": "files_index",
                "n_results": data.get("n_results", 10)
            }
        )
        return response.json()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 