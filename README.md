> Part of my **deployable AI systems portfolio**. See my [GitHub profile](https://github.com/EdwinEmesiani) for related projects like **HealthyLife Insurance Charge Prediction** and **Technoecom E-Commerce EDA**.

## 📄 RAG_10K_Report — Financial Intelligence Assistant

## 🔍 Overview

RAG_10K_Report is a Retrieval-Augmented Generation (RAG) system designed to answer financial questions directly from SEC 10-K filings with page-level citations.

The system combines semantic search, vector databases, and large language models to deliver grounded, traceable responses.

---

## 🏗 Architecture

1. 📄 10-K ingestion  
2. ✂️ Text chunking  
3. 🔢 Embedding using `thenlper/gte-large`  
4. 🗄 Chroma vector database persistence  
5. 🔎 Metadata-filtered semantic retrieval  
6. 🤖 GPT-powered answer generation  
7. 📚 Page citation formatting  
8. 🌐 Deployment via Gradio on Hugging Face Spaces  

---

## 🧰 Tech Stack

- Python
- LangChain
- ChromaDB
- Sentence Transformers
- OpenAI API
- Gradio
- Hugging Face Spaces
- Git & Git LFS

---

## 🔐 Security

- API keys handled via environment variables
- No secrets stored in repository
- HF Space secrets configured securely

---

## 🚀 Deployment

Deployed on Hugging Face Spaces using Gradio.

To run locally:

```bash
git clone https://github.com/yourusername/RAG_10K_Report.git
cd RAG_10K_Report
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
python app.py
