Retrieval-Augmented Generation (RAG) Q&A System
This project implements a Retrieval-Augmented Generation (RAG) system using Streamlit for the interface and the Gemini API for the Large Language Model (LLM). It is designed to answer user queries based on a fixed knowledge base, providing a confidence score for each response.

Setup
Install dependencies via pip install -r requirements.txt. Run the app with streamlit run rag_app.py.

Design Overview
This RAG system uses 'all-MiniLM-L6-v2' for embeddings and FAISS for vector search (Cosine Similarity).
The core logic separates document retrieval from answer generation, powered by the Gemini API.

Confidence Score
Confidence is the mean of the top $k$ cosine similarity scores from the retrieved knowledge chunks.

Future Improvement Note
If given more time, the system would be primarily improved by focusing on system design for 
production-readiness and scalability:

The immediate goal would be to implement an asynchronous ingestion and queuing system. This involves:

Adding a dedicated file attachment area in the Streamlit interface.

Triggering an asynchronous (background) process for PDF text extraction and embedding generation (vectorization).

Utilizing a message queue (like Redis or RabbitMQ) to manage document ingestion and retrieval requests. This ensures that multiple users can interact harmoniously and prevents one long ingestion process from blocking the entire application, leading to a much more responsive API. This architecture is necessary for handling concurrent users and large document uploads.