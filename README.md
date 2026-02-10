# Project Title: RAG-10K Financial Reports Q&A System

##  Overview
This project implements a **Retrieval-Augmented Generation (RAG)** system that enables financial analysts to ask natural-language questions and receive **grounded, citation-backed answers** directly from company **10-K filings**.

The system solves the problem of manually searching long financial documents by combining:
- Vector search over embedded 10-K reports
- Large Language Models (LLMs) for answer generation
- Source attribution (page-level citations)

The final solution is deployed as an **interactive Gradio application** on **Hugging Face Spaces**.

---

## 🎯 Objectives
- Build a **vector database** from multiple companies’ 10-K reports  
- Enable **semantic retrieval** of relevant document chunks  
- Generate **fact-grounded answers** using an LLM  
- Ensure responses are **traceable to original 10-K sources**  
- Deploy a **production-ready RAG application** for analyst use  

---

## 🧠 How the RAG System Works
1. **Ingestion:** 10-K PDFs are loaded, chunked, and embedded  
2. **Vector Storage:** Embeddings are persisted in a Chroma vector database  
3. **Retrieval:** Relevant chunks are retrieved based on the user query  
4. **Augmentation:** Retrieved context is injected into the LLM prompt  
5. **Generation:** The LLM produces an answer **only from retrieved context**  
6. **Citations:** Page numbers and document sources are included  

---

## 🧰 Tech Stack
| Category | Tools Used |
|-------|-----------|
| Language | Python 🐍 |
| RAG Framework | LangChain |
| Vector Database | ChromaDB |
| Embeddings | HuggingFace (gte-large) |
| LLM | OpenAI / AnyScale |
| Deployment | Gradio, Hugging Face Spaces |
| Version Control | Git & GitHub |

---

## 📊 Supported Companies
- Google  
- Microsoft  
- Amazon (AWS)  
- Meta  
- IBM  

Each company’s 10-K filing is indexed separately and queried dynamically.

---

## 🔍 Example Questions
- *Has the company made any significant acquisitions in the AI space?*  
- *How much capital has been allocated toward AI research and development?*  
- *What ethical AI initiatives has the company implemented?*  
- *How does the company differentiate itself in the AI market?*  

---

## 🖥️ Gradio Interface Features
- Company selector (radio button)  
- Natural-language question input  
- Adjustable retrieval depth (Top-K)  
- Citation-backed answers  
- Deployed as a **public Hugging Face Space**

---

## ⚙️ How to Run the Project

### Clone the repository
```bash
git clone https://github.com/< EdwinEmesiani >/< RAG_10k_report >.git
cd < RAG_10k_report >
