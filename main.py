import os
import io
import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Any, cast

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from pypdf import PdfReader
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai  # type: ignore
import jwt

# Load environment variables
load_dotenv()

# Initialize FastAPI App
app = FastAPI(title="AI Resume Analyzer API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SQLite Database Configuration
DATABASE_URL = "sqlite:///./resume_analyzer.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# SQLAlchemy Database Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

class Resume(Base):
    __tablename__ = "resumes"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String, nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)
    pinecone_namespace = Column(String, nullable=False)
    ats_score = Column(Integer, default=0)
    analysis_data = Column(Text, nullable=True)  # Store JSON string containing audit results
    raw_text = Column(Text, nullable=True)

# Create Database Tables
Base.metadata.create_all(bind=engine)

# Database Session Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# JWT Token Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "supersecretjwtkey123!@#change_me")
JWT_ALGORITHM = "HS256"

# Gemini model configuration (can be overridden in .env)
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash"))
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2")
PINECONE_INDEX_DIMENSION = int(os.getenv("PINECONE_INDEX_DIMENSION", "768"))

# Pydantic Schemas
class UserAuth(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str

class ChatRequest(BaseModel):
    question: str

# Password Hashing Utility (PBKDF2 SHA-256 - pure python, highly portable, avoids bcrypt compile errors on Windows)
def hash_password(password: str) -> str:
    salt = os.urandom(16)
    db_password = hashlib.pbkdf2_hmac(
        'sha256', 
        password.encode('utf-8'), 
        salt, 
        100000
    )
    return salt.hex() + ":" + db_password.hex()

def verify_password(password: str, hashed_password: str) -> bool:
    try:
        salt_hex, key_hex = hashed_password.split(":")
        salt = bytes.fromhex(salt_hex)
        key = bytes.fromhex(key_hex)
        new_key = hashlib.pbkdf2_hmac(
            'sha256', 
            password.encode('utf-8'), 
            salt, 
            100000
        )
        return new_key == key
    except Exception:
        return False

# JWT Helper Functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=7)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_access_token(token: str):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None

# Dependency to check auth token
def get_current_user_id(authorization: str = Header(None), db: Session = Depends(get_db)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authentication token is missing. Please log in.")
    try:
        token_type, token = authorization.split(" ")
        if token_type.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid token type.")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header format.")
        
    payload = decode_access_token(token)
    if not payload or "user_id" not in payload:
        raise HTTPException(status_code=401, detail="Session expired or invalid token. Please log in again.")
        
    user_id = payload["user_id"]
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User account not found.")
    return user_id

# RAG & API Service Setup
def check_api_keys():
    gemini_key = os.getenv("GEMINI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    
    if not gemini_key or gemini_key.startswith("your_"):
        raise HTTPException(
            status_code=400, 
            detail="Gemini API Key is not configured. Please enter a valid GEMINI_API_KEY in the .env file."
        )
    if not pinecone_key or pinecone_key.startswith("your_"):
        raise HTTPException(
            status_code=400, 
            detail="Pinecone API Key is not configured. Please enter a valid PINECONE_API_KEY in the .env file."
        )
    # Optional: ensure models are configured
    text_model = os.getenv("GEMINI_TEXT_MODEL", GEMINI_TEXT_MODEL)
    embed_model = os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL)
    if not text_model or text_model.startswith("your_"):
        raise HTTPException(
            status_code=400,
            detail="Gemini text model is not configured. Please set GEMINI_TEXT_MODEL in .env to a valid model name."
        )
    if not embed_model or embed_model.startswith("your_"):
        raise HTTPException(
            status_code=400,
            detail="Gemini embedding model is not configured. Please set GEMINI_EMBEDDING_MODEL in .env to a valid model name."
        )

# PDF Extraction Utility
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    pdf_file = io.BytesIO(pdf_bytes)
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# Text Chunking Utility
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    # Clean whitespace
    text = " ".join(text.split())
    if not text:
        return chunks
        
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
        if chunk_size - overlap <= 0:
            break
    return chunks

# Pinecone Index Management
def get_pinecone_client():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        return None
    try:
        return Pinecone(api_key=api_key)
    except Exception as e:
        print(f"Error initializing Pinecone client: {e}")
        return None

def get_or_create_index():
    pc = get_pinecone_client()
    if not pc:
        return None
    index_name = os.getenv("PINECONE_INDEX_NAME", "resume-analyzer")
    try:
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing_indexes:
            # Create serverless index with configured dimension
            pc.create_index(
                name=index_name,
                dimension=PINECONE_INDEX_DIMENSION,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        return pc.Index(index_name)
    except Exception as e:
        print(f"Error getting/creating Pinecone index: {e}")
        return None

# Vector Indexing
def index_resume_chunks(resume_id: int, user_id: int, chunks: List[str]):
    pc_index = get_or_create_index()
    if not pc_index:
        raise Exception("Failed to access Pinecone index. Check PINECONE_API_KEY or connection.")
        
    gemini_key = os.getenv("GEMINI_API_KEY")
    gen = cast(Any, genai)
    gen.configure(api_key=gemini_key)
    
    vectors = []
    namespace = f"user_{user_id}_resume_{resume_id}"
    batch_size = 20
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        try:
            response = gen.embed_content(
                model=os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL),
                content=batch_chunks,
                task_type="retrieval_document"
            )
            # Normalize possible response shapes
            if isinstance(response, dict):
                embeddings = response.get('embeddings') or response.get('embedding') or []
            else:
                embeddings = []
            # If single vector returned, wrap it
            if embeddings and isinstance(embeddings[0], (float, int)):
                embeddings = [embeddings]
            for j, emb_values in enumerate(embeddings):
                # Normalize embedding results
                if isinstance(emb_values, dict) and 'values' in emb_values:
                    values = emb_values['values']
                else:
                    values = emb_values
                    
                chunk_index = i + j
                vectors.append({
                    "id": f"chunk_{chunk_index}",
                    "values": values,
                    "metadata": {
                        "text": batch_chunks[j],
                        "resume_id": resume_id,
                        "chunk_index": chunk_index
                    }
                })
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise e
            
    try:
        pc_index.upsert(vectors=vectors, namespace=namespace)
    except Exception as e:
        print(f"Error upserting vectors: {e}")
        raise e

# Vector Querying
def query_resume_context(resume_id: int, user_id: int, question: str, top_k: int = 5) -> str:
    pc_index = get_or_create_index()
    if not pc_index:
        return ""
        
    gemini_key = os.getenv("GEMINI_API_KEY")
    gen = cast(Any, genai)
    gen.configure(api_key=gemini_key)
    
    try:
        response = gen.embed_content(
            model=os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL),
            content=question,
            task_type="retrieval_query"
        )
        # Extract single embedding vector
        if isinstance(response, dict):
            if 'embedding' in response and isinstance(response['embedding'], list):
                query_vector = response['embedding']
            elif 'embeddings' in response and isinstance(response['embeddings'], list):
                # take first
                q = response['embeddings'][0]
                query_vector = q['values'] if isinstance(q, dict) and 'values' in q else q
            else:
                query_vector = response.get('embedding', [])
        else:
            query_vector = []
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return ""
        
    namespace = f"user_{user_id}_resume_{resume_id}"
    try:
        query_response = pc_index.query(
            namespace=namespace,
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
        matches = query_response.get("matches", [])
        chunks = []
        for match in matches:
            text = match.get("metadata", {}).get("text", "")
            if text:
                chunks.append(text)
        return "\n\n---\n\n".join(chunks)
    except Exception as e:
        print(f"Error searching Pinecone: {e}")
        return ""

# Vector Deletion
def delete_resume_vectors(resume_id: int, user_id: int):
    pc_index = get_or_create_index()
    if not pc_index:
        return
    namespace = f"user_{user_id}_resume_{resume_id}"
    try:
        pc_index.delete(delete_all=True, namespace=namespace)
    except Exception as e:
        print(f"Error clearing vector namespace {namespace}: {e}")

# Gemini Analysis Logic
def analyze_resume_text(text: str) -> dict:
    gemini_key = os.getenv("GEMINI_API_KEY")
    gen = cast(Any, genai)
    gen.configure(api_key=gemini_key)
    
    model = gen.GenerativeModel(os.getenv("GEMINI_TEXT_MODEL", GEMINI_TEXT_MODEL))
    
    prompt = f"""
    You are a professional resume writer and ATS (Applicant Tracking System) optimization specialist.
    Analyze the following resume text and provide a comprehensive evaluation.
    
    Provide the response strictly as a JSON object with the following structure:
    {{
        "ats_score": 85,
        "strengths": [
            "Highlight of strength 1",
            "Highlight of strength 2"
        ],
        "issues": [
            "Description of resume issue or missing detail 1",
            "Description of resume issue or missing detail 2"
        ],
        "changes": [
            "Actionable recommendations to fix issues 1",
            "Actionable recommendations to fix issues 2"
        ],
        "career_paths": [
            "Recommended Job Title 1",
            "Recommended Job Title 2",
            "Recommended Job Title 3"
        ]
    }}

    Be critical but constructive. Base your evaluation on industry standards for ATS filtering (keywords, structure, project details).
    
    Resume Text:
    {text}
    """
    
    try:
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error running Gemini analysis: {e}")
        return {
            "ats_score": 50,
            "strengths": ["Loaded raw profile text successfully."],
            "issues": ["Could not perform deep API auditing on formatting."],
            "changes": ["Ensure your Gemini configuration limits and credit balance are healthy."],
            "career_paths": ["General Industry Specialist"]
        }

# Gemini RAG QA Logic
def answer_resume_question(resume_id: int, user_id: int, raw_text: str, question: str) -> str:
    context = query_resume_context(resume_id, user_id, question)
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    gen = cast(Any, genai)
    gen.configure(api_key=gemini_key)
    
    model = gen.GenerativeModel(os.getenv("GEMINI_TEXT_MODEL", GEMINI_TEXT_MODEL))
    
    # Fallback to passing a truncated portion of the raw text if context search yielded nothing
    fallback_text = raw_text[:3000] if raw_text else ""
    
    prompt = f"""
    You are an expert career assistant and AI resume assistant.
    You are helping a candidate evaluate, clarify, or refine details of their experience based on their uploaded resume.
    
    Here is some relevant context from their resume retrieved through semantic search (Pinecone RAG):
    ---
    {context if context.strip() else "No specific snippets matched. Truncated raw resume preview: " + fallback_text}
    ---
    
    Please answer the user's question. If the user asks general questions about career options, how to format things, or how to phrase a bullet point, offer helpful resume-writing tips. If the user asks about facts in the resume (e.g. 'What projects are listed?'), only list the details present in the context.
    Keep the answer clear, professional, and well-formatted.
    
    Question: {question}
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini model: {str(e)}"

# FASTAPI API ENDPOINTS

@app.post("/api/auth/register", response_model=TokenResponse)
def register(user_data: UserAuth, db: Session = Depends(get_db)):
    if not user_data.username.strip() or not user_data.password.strip():
        raise HTTPException(status_code=400, detail="Username and password cannot be empty.")
        
    db_user = db.query(User).filter(User.username == user_data.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username is already taken.")
    
    hashed_pw = hash_password(user_data.password)
    new_user = User(username=user_data.username, hashed_password=hashed_pw)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    token = create_access_token({"user_id": new_user.id})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/api/auth/login", response_model=TokenResponse)
def login(user_data: UserAuth, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user_data.username).first()
    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    hashed_pw = cast(str, db_user.hashed_password)
    if not verify_password(user_data.password, hashed_pw):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    
    token = create_access_token({"user_id": db_user.id})
    return {"access_token": token, "token_type": "bearer"}

@app.get("/api/auth/me")
def get_me(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return {"id": cast(int, user.id), "username": cast(str, user.username)}

@app.get("/api/resumes")
def list_resumes(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    resumes = db.query(Resume).filter(Resume.user_id == user_id).all()
    return [
        {
            "id": r.id,
            "filename": r.filename,
            "upload_date": r.upload_date.isoformat(),
            "ats_score": r.ats_score
        }
        for r in resumes
    ]

@app.get("/api/resumes/{resume_id}/analysis")
def get_resume_analysis(resume_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    resume = db.query(Resume).filter(Resume.id == resume_id, Resume.user_id == user_id).first()
    if not resume:
        raise HTTPException(status_code=404, detail="Resume record not found.")
    
    analysis = {}
    analysis_data_raw = cast(Optional[str], resume.analysis_data)
    if analysis_data_raw:
        try:
            analysis = json.loads(analysis_data_raw)
        except Exception:
            pass
            
    return {
        "id": resume.id,
        "filename": resume.filename,
        "ats_score": resume.ats_score,
        "analysis": analysis
    }

@app.post("/api/resumes/upload")
async def upload_resume(
    file: UploadFile = File(...),
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    check_api_keys()
    
    filename_str = cast(str, file.filename)
    if not filename_str.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    new_resume = None
    try:
        pdf_bytes = await file.read()
        text = extract_text_from_pdf(pdf_bytes)
        
        if not text.strip():
            raise HTTPException(
                status_code=400, 
                detail="No text could be extracted from this PDF. Please check if it's scanned or corrupted."
            )
            
        new_resume = Resume(
            user_id=user_id,
            filename=file.filename,
            pinecone_namespace="",
            ats_score=0,
            analysis_data="{}",
            raw_text=text
        )
        db.add(new_resume)
        db.commit()
        db.refresh(new_resume)
        
        # 1. Update Pinecone namespace setting
        namespace = f"user_{user_id}_resume_{cast(int, new_resume.id)}"
        setattr(new_resume, "pinecone_namespace", namespace)
        
        # 2. Run Gemini audit for ATS scoring and recommendations
        analysis = analyze_resume_text(text)
        setattr(new_resume, "ats_score", int(analysis.get("ats_score", 50)))
        setattr(new_resume, "analysis_data", json.dumps(analysis))
        db.commit()
        
        # 3. Process RAG chunks
        chunks = chunk_text(text)
        if chunks:
            index_resume_chunks(cast(int, new_resume.id), user_id, chunks)
            
        return {
            "id": new_resume.id,
            "filename": new_resume.filename,
            "ats_score": new_resume.ats_score,
            "analysis": analysis
        }
        
    except HTTPException as he:
        if new_resume and getattr(new_resume, "id", None):
            db.delete(new_resume)
            db.commit()
        raise he
    except Exception as e:
        if new_resume and getattr(new_resume, "id", None):
            db.delete(new_resume)
            db.commit()
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Indexing & processing failed: {str(e)}")

@app.delete("/api/resumes/{resume_id}")
def delete_resume(resume_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    resume = db.query(Resume).filter(Resume.id == resume_id, Resume.user_id == user_id).first()
    if not resume:
        raise HTTPException(status_code=404, detail="Resume record not found.")
        
    try:
        delete_resume_vectors(resume_id, user_id)
    except Exception as e:
        print(f"Error deleting Pinecone vectors: {e}")
        
    db.delete(resume)
    db.commit()
    return {"detail": "Resume deleted successfully."}

@app.post("/api/resumes/{resume_id}/chat")
def chat_with_resume(
    resume_id: int,
    request: ChatRequest,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    check_api_keys()
    
    resume = db.query(Resume).filter(Resume.id == resume_id, Resume.user_id == user_id).first()
    if not resume:
        raise HTTPException(status_code=404, detail="Resume record not found.")
        
    try:
        answer = answer_resume_question(resume_id, user_id, cast(str, resume.raw_text), request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate chatbot response: {str(e)}")

# Mount static folder
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")

# Startup check log
@app.on_event("startup")
def check_configuration():
    gemini_key = os.getenv("GEMINI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    
    issues = []
    if not gemini_key or gemini_key.startswith("your_"):
        issues.append("GEMINI_API_KEY is not defined in your environment/.env")
    if not pinecone_key or pinecone_key.startswith("your_"):
        issues.append("PINECONE_API_KEY is not defined in your environment/.env")
        
    if issues:
        print("\n" + "!" * 60)
        print("WARNING: Missing API keys in your configuration:")
        for issue in issues:
            print(f" - {issue}")
        print("Please create/update the .env file to enable processing resumes!")
        print("!" * 60 + "\n")
