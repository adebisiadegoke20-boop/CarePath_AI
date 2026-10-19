# CarePath AI: Multi-Agent Clinical Triage & Patient Assistant

**Developed by LOGIC LEAGUE TEAM**  
*NECA ICT Academy / ITF Advanced AI Automation Engineer Programme*

## About Our Project

This project was built by **Logic League Team** as part of our intensive learning journey in the **NECA ICT Academy / ITF Advanced AI Automation Engineer Programme**.

For this project, our team designed and deployed **CarePath AI**—a specialized multi-AI agent system engineered to provide automated clinical triage, maternal guidance, and appointment routing for the **Obstetrics & Gynecology (OB/GYN) Department at Garki Hospital Abuja**. 

Instead of relying on a single general-purpose AI model that risks hallucinations or generic advice, we created a modular multi-agent architecture where specialized agents handle distinct clinical and administrative tasks. The system grounds all medical advice strictly in verified healthcare guidelines (such as WHO Antenatal Care Protocols and Nigerian National Basic Newborn Care Guidelines) while enabling zero-barrier access via standard **Twilio SMS** for patients without internet connectivity or smartphones.

---

## What the Project Does

CarePath AI integrates several specialized components and agents to deliver a seamless, multi-channel triage experience:

- **Chatbot Orchestrator (`chatbot.py`)**: Classifies patient inquiries, manages context, and routes queries to the appropriate sub-agent or execution engine.
- **Clinical Triage Agent (`triage_engine.py`)**: Evaluates reported symptoms against WHO ANC and Nigerian clinical guidelines to perform preliminary risk assessment and flag obstetric emergencies (e.g., severe bleeding, pre-eclampsia signs).
- **Appointment Booking Agent (`appointment_booking.py`)**: Handles patient scheduling and directs mothers to appropriate OB/GYN clinic hours at Garki Hospital Abuja.
- **Patient Follow-Up & Reminders Agent (`reminders.py`)**: Manages routine care reminders, antenatal contact schedules, and immunization timelines.
- **Safety & Security Guardrails (`guard.py`)**: Validates patient input and filters generated outputs to reject prompt injections, block out-of-scope/malicious queries, and enforce strict clinical safety boundaries.
- **RAG Retrieval Engine (`carepath_rag_engine.py`)**: Executes vector similarity searches against Supabase `pgvector` to ground every response directly in validated medical PDFs.
- **Twilio SMS Webhook (`/sms`)**: Enables two-way, real-time SMS messaging so patients with basic phones can text questions and receive instant triage responses.

---

## Technologies Used

- **Programming Language**: Python 3.10+
- **Backend Framework**: FastAPI, Uvicorn
- **AI Orchestration & LLM Engine**: Groq API (`openai/gpt-oss-120b`)
- **Vector Database & Search**: Supabase with `pgvector` extension
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Telecommunications / SMS Gateway**: Twilio API & Webhooks
- **Data Extraction & Ingestion**: PyPDF / Custom Python extractors
- **Hosting & Deployment**: Render (Backend API), Bolt.new (Frontend Web Interface)

---

## Project Directory Structure

```text
Project 3- CAREPATH/
│
├── documents/
│   ├── processed/
│   └── raw/
│       ├── nigeria_national_guideline_basic_newborn_care.pdf
│       ├── tci_eight_anc_contacts_protocol.pdf
│       └── who_antenatal_care_guideline.pdf
│
├── .env
├── .gitignore
├── api.py                  # FastAPI endpoints & Twilio /sms webhook controller
├── appointment_booking.py  # OB/GYN clinic scheduling logic
├── carepath_rag_engine.py  # RAG retrieval pipeline & Groq LLM generation
├── chatbot.py              # Multi-agent orchestrator & query router
├── extract_clean.py        # PDF text parsing & vector chunking utility
├── guard.py                # Input/output security filters & clinical guardrails
├── reminders.py            # Antenatal appointment & care reminder protocols
├── requirements.txt        # Python package dependencies
├── runtime.txt             # Deployment python environment specification
└── triage_engine.py        # Clinical triage & symptom risk classification logic

**How It Works**
Submission: A user submits a query via the Bolt frontend or texts the Garki Hospital Twilio SMS line.

Safety Check & Routing: guard.py verifies the query safety, and chatbot.py routes the request to the appropriate agent (triage_engine, appointment_booking, or reminders).

Retrieval (RAG): carepath_rag_engine.py embeds the query using sentence-transformers and retrieves matching clinical context chunks from Supabase pgvector.

Grounded Generation: Groq generates a concise, clinically grounded response incorporating retrieved WHO/Nigerian guidelines and custom system prompts.

Response Delivery: The formatted assessment is delivered back to the frontend API or sent directly to the patient's mobile phone via Twilio SMS.

**Live Applications**
Backend API Health Check: https://carepath-ai-x8qu.onrender.com/
Interactive API Documentation (Swagger): https://carepath-ai-x8qu.onrender.com/docs
Twilio SMS Endpoint: https://carepath-ai-x8qu.onrender.com/sms
Bolt:

Frontend Web Application: Deploying via Bolt.new

Running the Project Locally
1. Clone the Repository
Bash
git clone [https://github.com/YOUR_USERNAME/carepath-ai.git]()
2. Create and Activate a Virtual Environment
Bash
python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
3. Install Required Dependencies
Bash
python -m pip install -r requirements.txt
4. Configure Environment Variables
Create a .env file in the root directory and configure your credentials:

Code snippet
GROQ_API_KEY=your_groq_api_key
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_KEY=your_supabase_service_role_key
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=your_twilio_phone_number
Environment Variables Security
Never commit your .env file, API keys, or database credentials to GitHub. Always ensure .env is properly declared in your .gitignore file.

**Learning Outcomes & Project Importance**
What I Learned Building This Project
Multi-Agent Orchestration: Gained hands-on experience designing modular AI workflows where specialized agents handle distinct domain tasks (triage, scheduling, reminders, and security) rather than relying on a monolithic prompt.

Production RAG Implementation: Learned how to connect vector embeddings with Supabase (pgvector) to anchor generative AI responses in verified medical literature, effectively mitigating model hallucinations.

API & Telecom Integration: Mastered building asynchronous backend APIs using FastAPI and integrating real-world communication webhooks (Twilio SMS) to bridge cloud AI with mobile text messaging.

Safety & Guardrails: Learned the importance of input and output filtering to enforce strict medical safety boundaries and block out-of-scope or malicious queries.

Cloud Deployment & Optimization: Developed practical skills in preparing, containerizing, and deploying Python-based machine learning applications to production cloud environments (Render).

**Importance of the Project**
Zero-Barrier Healthcare Access: By integrating Twilio SMS, CarePath AI ensures that expectant mothers and patients in low-resource or rural settings without smartphones or active internet data can still receive instant preliminary clinical guidance.

Clinical Safety & Grounding: Restricting AI reasoning strictly to official protocols (such as WHO Antenatal Care guidelines and Nigerian healthcare standards) ensures that patients receive reliable, evidence-based triage support.

Emergency Triage Efficiency: The system provides automated early detection of critical obstetric danger signs, ensuring timely referrals and routing patients directly to the 24-hour Garki Hospital Abuja OB/GYN emergency ward.