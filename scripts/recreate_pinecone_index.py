import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "resume-analyzer")
DIM = int(os.getenv("PINECONE_INDEX_DIMENSION", "768"))

if not API_KEY:
    raise SystemExit("PINECONE_API_KEY not set in environment/.env")

pc = Pinecone(api_key=API_KEY)

print(f"Existing indexes: {[i.name for i in pc.list_indexes()]}")

if INDEX_NAME in [i.name for i in pc.list_indexes()]:
    print(f"Deleting existing index '{INDEX_NAME}'...")
    try:
        pc.delete_index(INDEX_NAME)
        # wait a moment for deletion to propagate
        time.sleep(2)
        print("Deleted.")
    except Exception as e:
        print(f"Failed to delete index: {e}")

print(f"Creating index '{INDEX_NAME}' with dimension={DIM}...")
try:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("Index created.")
    print(f"Now available: {[i.name for i in pc.list_indexes()]}")
except Exception as e:
    print(f"Failed to create index: {e}")
    raise
