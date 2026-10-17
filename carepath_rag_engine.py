import os
import re
from dotenv import load_dotenv
from supabase import create_client
from groq import Groq

load_dotenv()

# ============================================================
# INITIALIZATION & LAZY CONNECTIONS
# ============================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in environment.")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Singleton for lazy loading heavy embedding model
_embedding_model = None

def get_embedding_model():
    """Lazy loads SentenceTransformer to prevent slow server startup & port timeouts."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model

print("System Initialized (Connected to Supabase & Groq)")


# ============================================================
# DOCUMENT PROCESSING & INGESTION
# ============================================================

def chunk_text(text, chunk_size=500, overlap=50):
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


def ingest_document(title, content, source):
    """Chunks text, computes local vector embeddings, and stores in carepath_documents."""
    print()
    print("Ingesting:", title)
    print("-" * 40)

    model = get_embedding_model()
    chunks = chunk_text(content)
    stored = 0

    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()

        try:
            supabase.table("carepath_documents").insert({
                "title": title,
                "content": chunk,
                "source": source,
                "page_number": i + 1,
                "embedding": embedding
            }).execute()

            stored += 1
        except Exception as e:
            print(f"Failed to insert chunk {i + 1}: {e}")

    print(f"Stored {stored} vectorized chunks from {title}")
    return stored


# ============================================================
# RAG SEARCH & INFERENCE
# ============================================================

def rag_search(question, top_k=5):
    """Queries vector embeddings via Supabase RPC function match_carepath_documents."""
    model = get_embedding_model()
    query_embedding = model.encode(question).tolist()

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
        print(f"Vector search failed: {err}")
        return []


def rag_generate(question, documents):
    """Generates structured response using Groq (openai/gpt-oss-120b)."""
    context = ""

    for i, doc in enumerate(documents):
        context += (
            "Source: " + doc.get("title", "Guideline") +
            " (chunk " + str(doc.get("page_number", 1)) + ")\n"
        )
        context += doc.get("content", "")

        if i < len(documents) - 1:
            context += "\n---\n"

    system_instruction = """
You are CarePath AI, a clinical assistant for the Obstetrics & Gynecology (OB/GYN) Department at Garki Hospital Abuja.
Answer questions based ONLY on the provided clinical documents (WHO ANC Guidelines, Nigeria Newborn Care, TCI Protocols).

Strict Operating Rules:
1. If the answer cannot be derived from the documents, respond: "I don't have enough information in the provided documents."
2. Never hallucinate or assume facts outside the provided document text.
3. Cite sources explicitly.
4. For emergencies, explicitly advise visiting the OB/GYN Department at Garki Hospital Abuja immediately.
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Documents:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.1
    )

    return response.choices[0].message.content


def full_rag(question):
    """Full execution pipeline for a user question."""
    print()
    print("Question:", question)
    print("-" * 50)

    # Simple Input Guardrail
    restricted = [r"\bhack\b", r"\bexploit\b", r"\bmalware\b"]
    if any(re.search(pat, question.lower()) for pat in restricted):
        print("Input rejected by CarePath safety guardrails.")
        return {
            "answer": "Query flagged by security rules.",
            "sources": []
        }

    print("Searching for relevant clinical guidelines...")
    docs = rag_search(question)

    if not docs:
        print("No relevant documents found.")
        return {
            "answer": "I don't have enough information in the provided documents.",
            "sources": []
        }

    print(f"Found {len(docs)} relevant chunks:")

    for doc in docs:
        print(
            "-",
            doc["title"],
            "(similarity:",
            round(doc["similarity"], 3),
            ")"
        )

    print()
    print("Generating response via Groq openai/gpt-oss-120b...")
    answer = rag_generate(question, docs)

    # Append CarePath Garki Hospital Disclaimer
    disclaimer = "\n\n[Disclaimer: CarePath provides preliminary triage based on WHO and Nigerian clinical protocols. For emergencies, report directly to the Obstetrics & Gynecology (OB/GYN) Department at Garki Hospital Abuja.]"
    final_answer = answer + disclaimer

    sources = []
    for doc in docs:
        sources.append({
            "title": doc["title"],
            "chunk": doc["page_number"],
            "similarity": round(doc["similarity"], 3)
        })

    print()
    print("Answer:")
    print(final_answer)

    print()
    print("Sources:")
    for source in sources:
        print(
            "-",
            source["title"],
            "(chunk",
            source["chunk"],
            ")"
        )

    return {
        "answer": final_answer,
        "sources": sources
    }


# ============================================================
# EXECUTING SAMPLE CLINICAL INGESTION & TESTS
# ============================================================

if __name__ == "__main__":
    print()
    print("Storing CarePath Sample Clinical Knowledge Base")
    print("-" * 50)

    anc_content = """
    WHO Antenatal Care Guidelines: Pregnant women are advised to start antenatal care contacts within the first trimester.
    The Nigerian guidelines support the TCI Eight ANC Contacts protocol to ensure comprehensive monitoring.
    Routine ANC visits involve measuring blood pressure, checking weight, checking hemoglobin levels, screening for pre-eclampsia, and evaluating fetal growth.
    Women should report any history of hypertension, diabetes, or prior delivery complications to their midwife or obstetrician.
    """

    warning_signs_content = """
    Obstetric Warning Signs Protocol: Urgent danger signs during pregnancy include heavy vaginal bleeding, severe epigastric or abdominal pain, high fever, fits or convulsions, fluid leakage, or severe persistent headache with blurred vision.
    Any pregnant patient experiencing severe warning signs must be transferred immediately to an emergency facility.
    At Garki Hospital Abuja, the OB/GYN ward provides 24-hour emergency triage and maternal management.
    """

    appointment_content = """
    Garki Hospital Abuja OB/GYN Appointment Rules: Appointments can be scheduled specifically for the Obstetrics & Gynecology Department.
    When booking, patients must provide their full name, phone number, estimated gestational age, and reason for consultation.
    Routine consultations are held for antenatal contacts, postnatal follow-ups, and gynaecological reviews.
    Patients experiencing acute emergency symptoms must bypass routine scheduling and report directly to emergency.
    """

    ingest_document(
        "WHO Antenatal Care Guideline",
        anc_content,
        "who_anc_guideline.txt"
    )

    ingest_document(
        "Obstetric Warning Signs Protocol",
        warning_signs_content,
        "warning_signs.txt"
    )

    ingest_document(
        "Garki Hospital OB/GYN Appointment Policy",
        appointment_content,
        "garki_appointments.txt"
    )

    print()
    print("=" * 50)
    print("CAREPATH MULTI-AGENT RAG TESTS")
    print("=" * 50)

    full_rag("When should a pregnant patient begin antenatal contacts and what checks are conducted?")
    full_rag("What dangerous symptoms require immediate hospital triage?")
    full_rag("How do I schedule an appointment with the OB/GYN department at Garki Hospital Abuja?")
    full_rag("A pregnant patient has severe abdominal pain and heavy bleeding. Should they book a routine appointment?")
    full_rag("What hospital in Abuja provides free antenatal care?")