# 🧠 Retrieval-Augmented Generation (RAG) Q&A System

This project implements a **Retrieval-Augmented Generation (RAG)** system using **Streamlit** for the user interface and the **Gemini API** as the Large Language Model (LLM).  
It is designed to answer user queries based on a fixed knowledge base, providing **accurate responses** with an associated **confidence score**.

---

## 🚀 Features

- **Interactive Streamlit Interface** for real-time Q&A.  
- **RAG Pipeline** that retrieves context before generating an answer.  
- **Embeddings with `all-MiniLM-L6-v2`** for semantic search.  
- **FAISS Vector Store** for efficient similarity-based document retrieval.  
- **Confidence Scoring System** to indicate response reliability.

---

## ⚙️ Setup Instructions

1. Install Dependencies
pip install -r requirements.txt
2. Run the Streamlit App
bash
Copy code
streamlit run rag_app.py

---

## 🧩 Design Overview
The RAG system follows a modular architecture with clear separation between document retrieval and answer generation.

🔹 Embedding Model
Uses all-MiniLM-L6-v2 to convert documents and queries into vector embeddings.

🔹 Vector Store
Employs FAISS (Facebook AI Similarity Search) for fast and scalable cosine similarity-based search.

🔹 LLM Integration
The Gemini API is used for generating final answers after retrieving the most relevant context.

---

## 📊 Confidence Score Calculation

The confidence score represents how relevant the retrieved context is to the user query.

Confidence=Mean of Top-k Cosine Similarity Scores

This score provides a transparent measure of how much the model “trusts” its retrieved data.

---

## 🔮 Future Improvements
If given additional time, the system can be enhanced for production readiness and scalability with the following improvements:

🧵 1. Asynchronous Ingestion & Queuing
Implement a background ingestion system for large documents.

Allow users to upload PDFs without blocking the main thread.

🗃️ 2. Message Queue Integration
Use Redis or RabbitMQ to manage ingestion and retrieval requests.

Prevent long-running processes from blocking the application.

Support multiple concurrent users smoothly.

⚙️ 3. Enhanced Architecture
Introduce an asynchronous pipeline for vectorization and retrieval.

Enable scalable deployment through containerization or microservices.

Vector Store	FAISS
Backend	Python
Task Queue (Future)	Redis / RabbitMQ

---

## 🧠 Architecture Diagram
                ┌────────────────────────────┐
                │        User Query          │
                └────────────┬───────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │     Streamlit UI     │
                  └────────────┬─────────┘
                             │
                             ▼
              ┌────────────────────────────────┐
              │   Embedding Model (MiniLM-L6)   │
              └────────────────┬────────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
      ┌─────────────────┐           ┌─────────────────────┐
      │ Knowledge Base   │  ----->  │  FAISS Vector Store  │
      └─────────────────┘           └─────────────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │   Top-k Relevant Chunks   │
                  └────────────┬─────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │   Gemini LLM (Answer)    │
                  └────────────┬─────────────┘
                               │
                               ▼
                     ┌──────────────────┐
                     │  Final Response  │
                     └──────────────────┘

                     
---

## 🧑‍💻 Author

**Harikrishnan V**  
AI Engineer | Machine Learning & RAG Systems Developer  

🔗 [LinkedIn](https://www.linkedin.com/in/harivansu)  
💻 [GitHub](https://github.com/yourusername)

