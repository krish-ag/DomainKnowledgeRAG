# schemas.py

from pydantic import BaseModel, ConfigDict, EmailStr

# =====================

# USER SCHEMAS

# =====================

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

# =====================

# CORPUS SCHEMAS

# =====================

class CorpusCreate(BaseModel):
    name: str

class CorpusResponse(BaseModel):
    id: int
    name: str

    model_config = ConfigDict(from_attributes=True)

# =====================

# DOCUMENT SCHEMAS

# =====================

class DocumentResponse(BaseModel):
    id: int
    filename: str

    model_config = ConfigDict(from_attributes=True)

# =====================

# QUERY SCHEMAS

# =====================

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# =====================

# CHAT SCHEMAS

# =====================

class ChatMessageResponse(BaseModel):
    id: int
    question: str
    answer: str

    model_config = ConfigDict(from_attributes=True)

# =====================

# GENERIC RESPONSE

# =====================

class MessageResponse(BaseModel):
    message: str


