import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

# ============================================================
# INITIALIZATION & CONNECTIONS
# ============================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise ValueError("Check SUPABASE_URL and SUPABASE_SERVICE_KEY in your .env file.")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=GROQ_API_KEY)

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "documents" / "processed"

print("CarePath RAG Engine Initialized")
print("Connected to Supabase PostgreSQL")
print("Embedding Model Loaded (all-MiniLM-L6-v2)")
print("Groq Client Active (openai/gpt-oss-120b)")


# ============================================================
# DOCUMENT PROCESSING & VECTOR INGESTION
# ============================================================

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Splits full clinical guidelines into overlapping text chunks."""
    text = text.replace("\x00", "")  # Sanitize null bytes for Postgres
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            last_period = text.rfind(".", start, end)
            if last_period > start + chunk_size // 2:
                end = last_period + 1

        chunk = text[start:end].strip()

        if len(chunk) > 20:
            chunks.append(chunk)

        new_start = end - overlap
        if new_start <= start:
            new_start = end

        start = new_start

    return chunks


def ingest_document(title: str, content: str, source: str) -> int:
    """Chunks text, batch computes vector embeddings, and bulk inserts into carepath_documents."""
    print(f"\nIngesting: {title}")
    print("-" * 50)

    chunks = chunk_text(content)
    if not chunks:
        print(f"No chunks found for {title}")
        return 0

    print(f"Encoding {len(chunks)} chunks in bulk (this may take a moment)...")
    # Batch encode all chunks at once (much faster than loop encoding)
    embeddings = embedding_model.encode(chunks, show_progress_bar=True).tolist()

    rows_to_insert = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        rows_to_insert.append({
            "title": title,
            "content": chunk,
            "source": source,
            "page_number": i + 1,
            "embedding": embedding
        })

    # Bulk insert into Supabase in batches of 50 to avoid network timeouts & payload limits
    stored = 0
    batch_size = 50
    for j in range(0, len(rows_to_insert), batch_size):
        batch = rows_to_insert[j:j + batch_size]
        try:
            supabase.table("carepath_documents").insert(batch).execute()
            stored += len(batch)
            print(f"Inserted batch {j // batch_size + 1} ({stored}/{len(rows_to_insert)} chunks)...")
        except Exception as e:
            print(f"Failed to insert batch: {e}")

    print(f"Successfully stored {stored} vectorized chunks from {title}")
    return stored

    print(f"Stored {stored} vectorized chunks from {title}")
    return stored


def ingest_from_processed_directory():
    """Reads all clean text files from documents/processed/ and uploads vector embeddings."""
    files_to_upload = [
        {"title": "WHO Recommendations on Antenatal Care", "filename": "who_anc_guideline.txt", "source": "who_anc_guideline"},
        {"title": "Nigeria Guideline for Basic Newborn Care", "filename": "nigeria_basic_newborn_care.txt", "source": "nigeria_basic_newborn_care"},
        {"title": "The Challenge Initiative ANC Protocol", "filename": "tci_anc_eight_contacts.txt", "source": "tci_anc_eight_contacts"}
    ]

    for file_info in files_to_upload:
        file_path = PROCESSED_DIR / file_info["filename"]
        if not file_path.exists():
            print(f"[Warning] File not found: {file_path.name}. Run extract_clean.py first.")
            continue

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            full_text = f.read()

        ingest_document(
            title=file_info["title"],
            content=full_text,
            source=file_info["source"]
        )


# ============================================================
# VECTOR SEARCH & CLINICAL INFERENCE
# ============================================================

def rag_search(question: str, top_k: int = 5) -> list[dict]:
    """Queries vector embeddings via Supabase RPC function match_carepath_documents."""
    query_embedding = embedding_model.encode(question).tolist()

    try:
        results = supabase.rpc(
            "match_carepath_documents",
            {
                "query_embedding": query_embedding,
                "match_threshold": 0.0,
                "match_count": top_k
            }
        ).execute()

        return results.data or []
    except Exception as err:
        print(f"Vector search RPC execution failed: {err}")
        return []


def rag_generate(question: str, documents: list[dict]) -> str:
    """Generates clinical response using Groq (openai/gpt-oss-120b)."""
    context = ""

    for i, doc in enumerate(documents):
        context += (
            f"Source: {doc.get('title', 'Guideline')} "
            f"(Chunk {doc.get('page_number', 1)})\n"
        )
        context += f"{doc.get('content', '')}\n"

        if i < len(documents) - 1:
            context += "\n---\n"

    system_instruction = """
You are CarePath AI, a specialized clinical triage assistant for the Obstetrics & Gynecology (OB/GYN) Department at Garki Hospital Abuja.
Answer patient and clinical questions based strictly on the provided WHO and Nigerian national clinical guidelines.

Operating Directives:
1. Ground every statement in the retrieved guideline text.
2. If the guidelines do not contain enough information to answer, state: "I don't have enough information in the provided documents."
3. Never invent protocols or hallucinate clinical procedures.
4. For emergencies (e.g., severe bleeding, severe abdominal pain, fluid leak, convulsions), instruct the patient to report immediately to the OB/GYN Emergency Unit at Garki Hospital Abuja.
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Guideline Documents:\n{context}\n\nPatient Query: {question}"}
        ],
        temperature=0.1
    )

    return response.choices[0].message.content


def full_rag(question: str) -> dict:
    """End-to-end execution pipeline for user queries."""
    print()
    print("Question:", question)
    print("=" * 60)

    # Input Guardrail
    restricted_patterns = [r"\bhack\b", r"\bexploit\b", r"\bmalware\b"]
    if any(re.search(pattern, question.lower()) for pattern in restricted_patterns):
        print("Input blocked by CarePath security guardrails.")
        return {"answer": "Query flagged by safety rules.", "sources": []}

    print("Executing similarity vector search...")
    docs = rag_search(question)

    if not docs:
        print("No matching clinical documents found.")
        return {
            "answer": "I don't have enough information in the provided documents.",
            "sources": []
        }

    print(f"Retrieved {len(docs)} matching chunks:")
    for doc in docs:
        sim = round(float(doc.get("similarity", 0.0)), 4)
        print(f" - {doc['title']} (Chunk {doc['page_number']}) | Cosine Similarity: {sim}")

    print("\nGenerating grounded answer via Groq (openai/gpt-oss-120b)...")
    raw_answer = rag_generate(question, docs)

    disclaimer = (
        "\n\n[Clinical Notice: CarePath provides preliminary triage guidance based on "
        "WHO and Nigerian National Guidelines. For urgent medical concerns, report directly "
        "to the Obstetrics & Gynecology (OB/GYN) Department at Garki Hospital Abuja.]"
    )
    final_answer = raw_answer + disclaimer

    sources = []
    for doc in docs:
        sources.append({
            "title": doc["title"],
            "chunk": doc["page_number"],
            "similarity": round(float(doc.get("similarity", 0.0)), 4)
        })

    print("\nAnswer:")
    print(final_answer)

    return {
        "answer": final_answer,
        "sources": sources
    }


# ============================================================
# EXECUTION ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    # 1. Option to ingest processed files if they exist
    ingest_from_processed_directory()

    # 2. Run test queries
    print("\n" + "=" * 60)
    print("RUNNING CAREPATH CLINICAL RAG ENGINE EVALUATION")
    print("=" * 60)

    full_rag("When should a pregnant patient begin antenatal contacts and what checks are conducted?")
    full_rag("What dangerous symptoms require immediate hospital triage?")
    full_rag("How do I schedule an appointment with the OB/GYN department at Garki Hospital Abuja?")
    full_rag("A pregnant patient has severe abdominal pain and heavy bleeding. Should they book a routine appointment?")