from typing import Optional, Dict
from pydantic import BaseModel

class UserSession(BaseModel):
    session_id: str
    current_model: Optional[str] = None
    llm: Optional[any] = None
    loaded_model_config: Optional[dict] = None

    def cleanup(self):
        if self.llm:
            try:
                self.llm.stop()
            except:
                pass
            self.llm = None

    class Config:
        arbitrary_types_allowed = True

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, UserSession] = {}

    def create_session(self, session_id: str) -> UserSession:
        session = UserSession(session_id=session_id)
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[UserSession]:
        return self.sessions.get(session_id)

    def delete_session(self, session_id: str):
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.cleanup()
            del self.sessions[session_id] 