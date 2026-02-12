"""Build and persist a Chroma vector DB from 10-K PDFs.

Usage:
  python build_vectordb.py --data_dir ./Dataset-10k --persist_dir ./vector_db

Requires:
  - langchain, langchain-community, langchain-text-splitters
  - chromadb, pymupdf
  - sentence-transformers + torch
"""

import argparse
import os
import shutil
from pathlib import Path

import torch
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def company_from_filename(path: Path) -> str:
    name = path.stem.lower()
    if "ibm" in name:
        return "IBM"
    if "meta" in name:
        return "Meta"
    if "aws" in name or "amazon" in name:
        return "Amazon (AWS)"
    if "google" in name or "alphabet" in name:
        return "Google"
    if "msft" in name or "microsoft" in name:
        return "Microsoft"
    return path.stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--persist_dir", type=str, default="./vector_db")
    ap.add_argument("--collection", type=str, default="finsights_10k_gte_large")
    ap.add_argument("--chunk_size", type=int, default=1000)
    ap.add_argument("--chunk_overlap", type=int, default=150)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {data_dir.resolve()}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    all_docs = []
    for pdf_path in pdf_files:
        loader = PyMuPDFLoader(str(pdf_path))
        docs = loader.load()
        company = company_from_filename(pdf_path)
        for d in docs:
            d.metadata["company"] = company
            d.metadata["source"] = pdf_path.name
        all_docs.extend(docs)

    chunks = splitter.split_documents(all_docs)
    print(f"Loaded {len(all_docs)} pages; created {len(chunks)} chunks.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="thenlper/gte-large",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("Embeddings device:", device)

    if os.path.exists(args.persist_dir):
        shutil.rmtree(args.persist_dir)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=args.collection,
        persist_directory=args.persist_dir,
    )
    vectordb.persist()
    print("Persisted vector DB to:", os.path.abspath(args.persist_dir))


if __name__ == "__main__":
    main()
