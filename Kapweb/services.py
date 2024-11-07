import httpx
from fastapi import HTTPException
from typing import Any, Dict, Optional
import os
storage_host= "http://storage:8000"
storage_url = f"{storage_host}/v1/storage"
class ServiceHelper:
    def __init__(self, service_name: str):
        self.service_name = service_name
        
    async def store_data(self, key: str, value: Any, collection: str = "default") -> Dict:
        """Store data using the storage service"""
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{storage_url}/set", 
                json={
                    "key": key,
                    "value": value,
                    "collection": collection
                }
            )
            return response.json()

    async def get_data(self, key: str, collection: str = "default") -> Any:
        """Get data from the storage service"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{storage_url}/get/{collection}/{key}")
            return response.json()["value"]

    async def check_storage_ready(self) -> bool:
        """Check if storage service is ready"""
        try:
            # async with httpx.AsyncClient() as client:
            #     response = await client.get(f"{storage_host}/ready", timeout=2.0)
            #     return response.status_code == 200
            return False
        except:
            return False

    async def check_ready(self, dependencies: Dict[str, callable] = None) -> Dict:
        """
        Check if service and its dependencies are ready
        
        Args:
            dependencies: Dict of dependency name and check function
            Example: {
                "redis": redis_client.ping,
                "tesseract": pytesseract.get_tesseract_version
            }
        """
        try:
            # Vérifie que le service storage est accessible
            if not await self.check_storage_ready():
                raise HTTPException(status_code=503, detail="Storage service not ready")
            
            # Vérifie les dépendances spécifiques au service
            if dependencies:
                for name, check_func in dependencies.items():
                    try:
                        check_func()
                    except Exception as e:
                        raise HTTPException(
                            status_code=503, 
                            detail=f"Dependency {name} not ready: {str(e)}"
                        )
            
            return {"status": "ready", "service": self.service_name}
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=503, 
                detail=f"Service {self.service_name} not ready: {str(e)}"
            ) 