# RAG Assignments for InMind.ai

**Short description:**  
A retrieval-augmented generation (RAG) assignments project that explores document ingestion, chunking, embedding generation, retrieval, voice-based interaction, and manual evaluation workflows.

**Prepared by:** Joud Senan  
**Organization:** inmind.ai  
**Guidance:** Dani Azzam

---

## Overview

This repository contains my work for the RAG assignments at **inmind.ai**. The project focuses on building and evaluating a Retrieval-Augmented Generation pipeline that can ingest documents, generate embeddings, retrieve relevant context, and support evaluation workflows, including voice and manual evaluation components.

The implementation is intended to help explore how modern RAG systems are structured, tested, and improved in practical settings.

---

## Main Objectives

- Build a working RAG pipeline
- Process and chunk source documents
- Generate embeddings for semantic retrieval
- Store and retrieve vector representations efficiently
- Evaluate retrieval and response quality
- Experiment with voice-based and manual evaluation flows

---

## Project Components

The repository includes:

- **Document ingestion** for loading source files
- **Text chunking** for splitting long documents into manageable parts
- **Embedding generation** using sentence-transformer models
- **Vector storage / retrieval** for similarity search
- **LLM response generation** based on retrieved context
- **Voice workflow** for spoken interaction or voice-based input/output
- **Manual evaluation** scripts to inspect answer quality and retrieval relevance

---

## Typical Workflow

1. Load the input documents
2. Split the documents into chunks
3. Generate embeddings for each chunk
4. Store embeddings in a vector database or index
5. Retrieve the most relevant chunks for a query
6. Use the retrieved context to support answer generation
7. Evaluate the outputs manually or through the provided scripts

---

## Requirements

Make sure you have Python installed, preferably **Python 3.10+**.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python 01_react_rag_agent.py
python 02_multi_agent_sql_guardrails.py
python 03_voice_and_manual_eval.py --mode eval
python 03_voice_and_manual_eval.py --mode mic --seconds 8 --play
```

---

## Repository Structure

```text
.
├── data_agro_dummy/
├── agronomy_agent.db
├── manual_eval_results.csv
├── mic_question.wav
├── 01_react_rag_agent.py
├── 02_multi_agent_sql_guardrails.py
├── 03_voice_and_manual_eval.py
├── rag_core.py
├── voice_answer.wav
├── sample_chat_output.md
├── technical_justifications.md
├── requirements.txt
└── README.md
```

---

## What I Learned

Through these assignments, I practiced:

- structuring a RAG pipeline end to end
- working with embeddings and retrieval systems
- understanding chunking and indexing strategies
- evaluating retrieval quality
- exploring practical LLM application workflows
