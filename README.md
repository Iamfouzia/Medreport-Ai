# AI-Powered Medical Report Analysis Dashboard

> *Upload your report. Understand your health instantly.* 

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red)](https://your-live-link-here)
[![Python](https://img.shields.io/badge/Python-3.10-green)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.20-blue)](https://langchain.com)

---

## Overview

MedReport AI is an AI-powered medical report analysis dashboard. Upload a PDF or DOCX medical report and instantly get blood parameter visualization, severity assessment, and doctor referral suggestions  all powered by a multi-agent RAG pipeline.

Most patients receive medical reports they cannot understand. MedReport AI bridges that gap.

---

## Live Demo

🔗 [https://huggingface.co/spaces/foziakahn/MindEase-Mental-Health]

---

## Features

- 📄 **Multi-format Report Upload** →  Supports PDF, DOCX, and XLSX medical reports
- 📊 **Blood Parameter Visualization** → 3 interactive Plotly charts: range comparison, bar chart, and status overview
- 🤖 **Multi-Agent AI Pipeline** → 3 specialized agents: Report Agent, Severity Agent, and Referral Agent
- 🔍 **RAG-Based Q&A** → Ask questions about your report using FAISS vector search + LLM
- 🚨 **Severity Assessment** → Automatically classifies report as LOW, MEDIUM, or HIGH severity
- 👨‍⚕️ **Doctor Referral Suggestions** → Recommends which specialist to visit and how urgently
- 🩺 **General Medical Q&A** → Ask any medical question without uploading a report
- 📈 **Token Usage Tracking** → Real-time input/output token monitoring

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit 1.32 |
| LLM | Llama 3.1-8b-instant via Groq API |
| RAG Framework | LangChain + LangGraph |
| Vector Store | FAISS (faiss-cpu) |
| Embeddings | HuggingFace MiniLM-L6-v2 |
| Document Parsing | PyPDF2, python-docx, openpyxl |
| Visualization | Plotly |
| Environment | python-dotenv |

---

## System Architecture

```
User Uploads Report (PDF / DOCX / XLSX)
            │
            ▼
    Text Extraction
    (PyPDF2 / python-docx / openpyxl)
            │
            ▼
    Text Chunking
    (split_text_into_chunks)
            │
            ▼
    Vector Embeddings
    (HuggingFace MiniLM-L6-v2)
            │
            ▼
    FAISS Vector Store
            │
     ┌──────┴──────┐
     ▼             ▼
User Question   Auto Analysis
     │             │
     ▼             ▼
RAG Retrieval  ┌─────────────────┐
(Top-K Chunks) │  3 AI Agents    │
     │         │                 │
     ▼         │ 1. Report Agent │
  LLM Answer   │ 2. Severity     │
  (Groq)       │ 3. Referral     │
               └─────────────────┘
                       │
                       ▼
            Dashboard Results + Charts
```

---

## Multi-Agent Pipeline

The system uses **3 specialized LangChain agents**, each with its own prompt and responsibility:

**Agent 1  Report Agent**
→ Answers user questions based on retrieved report context
→ Uses RAG: FAISS retrieval 
→ Groq LLM
→ Returns plain English answers without assumptions beyond the report

**Agent 2  Severity Agent**
→ Reads extracted report context

→ Classifies severity as LOW, MEDIUM, or HIGH

→ Explains abnormal values in 2-3 simple sentences

**Agent 3  Referral Agent**
→Takes report context + severity level as input

→ Suggests which specialist to visit

→ Recommends urgency level: immediate, within a week, or routine checkup

→ Provides home care tips in the meantime

---

## Blood Parameter Detection

The system automatically extracts and evaluates **21 blood parameters** from reports:

| Category | Parameters |
|---|---|
| CBC | Hemoglobin, WBC, Platelet, RBC, Hematocrit, MCV, MCH, MCHC |
| Differential | Neutrophils, Lymphocytes |
| Diabetes | Fasting Blood Sugar, HbA1c |
| Liver Function | SGPT, SGOT, Bilirubin |
| Kidney Function | Creatinine, Urea |
| Nutrition | Iron, Ferritin, Vitamin D, Vitamin B12 |

Each parameter is compared against standard reference ranges and flagged as Normal, High, or Low.

---

## Visualization

Three interactive Plotly charts are generated automatically:

**Range Comparison Chart**  Diamond markers showing each value relative to its normal range

**Bar Chart**  Side-by-side values colored green (normal) or red (abnormal)

**Status Overview (Donut Chart)**  Summary of normal vs abnormal parameter count

---

## Project Structure

```
MedReport-AI/
│
├── app.py                      # Main Streamlit app, UI, charts, session state
├── requirements.txt            # Python dependencies
├── .env                        # API keys (not committed)
│
├── agents/
│   ├── report_agent.py         # Answers questions from report context
│   ├── severity_agent.py       # Classifies report severity
│   └── referral_agent.py       # Suggests doctor referral
│
└── utils/
    ├── pdf_processor.py        # Text extraction and chunking
    └── rag_pipeline.py         # FAISS vector store creation and search
```

---

## Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/Iamfouzia/MedReport-AI.git
cd MedReport-AI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file
echo "GROQ_API_KEY=your_groq_api_key_here" > .env

# 4. Run the app
streamlit run app.py
```

App will open at `http://localhost:8501`

---

## How to Use

1. Upload a PDF or DOCX medical report from the sidebar
2. View automatically extracted blood parameters and charts
3. Type a question in the chat box and click Analyze
4. Switch to General Q&A mode for medical questions without a report
5. View severity assessment and doctor referral in the results panel

---

## Disclaimer

MedReport AI is an AI assistant for informational purposes only. It is **not a substitute for professional medical advice**. Always consult a qualified physician before making any health decisions.

