from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from Kapweb.files import file_manager, StreamResponse, FileCallbacks
from Kapweb.services import ServiceHelper
from typing import List, Optional
import uuid
from datetime import datetime

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
    files: List[UploadFile] = File(...),
    path: str = Form(...),
):
    results = []
    for file in files:
        content = await file.read()
        file_path = f"{path}/{file.filename}"
        result = await file_manager.save_file(file_path, content)
        results.append(result)
    
    return JSONResponse(content={"uploads": results})

@app.post("/v1/files/move")
async def move_file(request: Request):
    data = await request.json()
    if not all(k in data for k in ["source", "destination"]):
        raise HTTPException(status_code=400, detail="Source et destination requises")
    
    result = await file_manager.move_file(data["source"], data["destination"])
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

@app.post("/v1/files/directory/compress")
async def compress_directory(request: Request):
    data = await request.json()
    if "path" not in data:
        raise HTTPException(status_code=400, detail="Chemin requis")
    
    async def stream_response():
        async for response in file_manager.compress_directory(
            data["path"],
            data.get("zip_name")
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
        
        yield 'data: [DONE]\n\n'

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 