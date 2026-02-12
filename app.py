import os
from pathlib import Path
import gradio as gr
import torch
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

COLLECTION_NAME = "finsights_10k_gte_large"
PERSIST_DIR = "./vector_db"

ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-4o-mini")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("ANYSCALE_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("ANYSCALE_BASE_URL")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "Missing API key. Set OPENAI_API_KEY (or ANYSCALE_API_KEY) as an environment secret."
    )

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,   # None is fine if not using a custom base_url
)


    


device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = HuggingFaceEmbeddings(
    model_name="thenlper/gte-large",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)

vectordb = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
)

SYSTEM_PROMPT = (
    "You are a careful financial analyst assistant. "
    "Answer using ONLY the provided context from 10-K filings. "
    "If the answer is not in the context, say you could not find it. "
    "Always cite page numbers in the form (p. X)."
)

def retrieve_context(query: str, company: str, k: int = 4):
    """
    Retrieve top-k relevant chunks for a query, filtered by company.
    Compatible with both old and new LangChain retriever APIs.
    """

    # Use existing vectordb (do NOT recreate it)
    retriever = vectordb.as_retriever(
        search_kwargs={"k": int(k), "filter": {"company": company}}
    )

    # New LangChain retriever API (>=0.1.x / 0.2.x)
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)

    # Older LangChain fallback
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(query)

    # Defensive fail (should never happen)
    raise RuntimeError(
        "Unsupported LangChain retriever version: "
        "no invoke() or get_relevant_documents() found."
    )
def format_context(docs):
    parts = []
    for i, d in enumerate(docs, start=1):
        page = d.metadata.get("page", None)
        src = d.metadata.get("source", "unknown")
        company = d.metadata.get("company", "unknown")
        header = f"[{i}] Company: {company} | Source: {src} | Page: {page}"
        parts.append(header + "\n" + d.page_content)
    return "\n\n".join(parts)

def build_messages(question: str, context: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:"},
    ]

def predict(company: str, question: str, top_k: int):
    if not question or not question.strip():
        return "Please enter a question."
    docs = retrieve_context(question, company=company, k=int(top_k))
    context = format_context(docs)
    resp = client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=build_messages(question, context),
        temperature=0.2,
    )
    answer = resp.choices[0].message.content
    # Append sources at end
    sources = "\n".join([f"- {d.metadata.get('source')} (p. {d.metadata.get('page')})" for d in docs])
    return f"{answer}\n\nSources:\n{sources}"

companies = ["Amazon (AWS)", "Google", "Microsoft", "Meta", "IBM"]

with gr.Blocks(title="Finsights Grey — 10-K RAG Q&A") as demo:
    gr.Markdown(
        """
        # 📄 Finsights Grey — 10-K RAG Q&A
        Select a company, ask a question, and get an answer grounded in the 10-K report.
        """
    )

    with gr.Row():
        company_in = gr.Radio(choices=companies, value="Google", label="Select Company")
        top_k = gr.Slider(1, 8, value=4, step=1, label="Top-K retrieved chunks")

    question_in = gr.Textbox(lines=3, label="Your question")
    answer_out = gr.Textbox(lines=14, label="Answer")

    gr.Button("Ask").click(fn=predict, inputs=[company_in, question_in, top_k], outputs=[answer_out])

if __name__ == "__main__":
    demo.launch()

