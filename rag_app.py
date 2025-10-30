import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please check your .env file or environment variables.")
try:    
    gemini_client = genai.Client(api_key=API_KEY)
    
except ValueError as e:
    st.error(str(e))
    st.stop()

# Knowledge Base (Mini Corpus) 
DOCUMENTS = {
    "Document_A_Mars_Colony": "The first human colony on Mars will likely face challenges with radiation shielding and growing food in Martian soil, known as regolith. The current plan involves using inflatable habitats before permanent underground structures can be built.",
    "Document_B_AI_Safety_Ethics": "AI Safety focuses on preventing autonomous systems from causing harm. A key ethical debate is the 'Trolley Problem' as applied to self-driving cars, where an immediate, difficult decision must be made to minimize overall damage.",
    "Document_C_Renewable_Energy": "Solar power and wind energy are leading the transition to renewables. A major challenge is energy storage, which is currently addressed by large-scale lithium-ion battery banks, but better solutions like flow batteries are under research.",
    "Document_D_Quantum_Computing": "Quantum computers use qubits, which can exist in a superposition of states, allowing them to solve problems that are intractable for classical computers. Shor's algorithm, for example, could theoretically break current public-key cryptography.",
    "Document_E_Deep_Learning_Basics": "Deep learning is a subfield of machine learning that uses neural networks with multiple layers (hence 'deep'). Key architectures include Convolutional Neural Networks (CNNs) for image processing and Recurrent Neural Networks (RNNs) for sequence data.",
    "Document_F_Black_Holes": "A black hole is a region of spacetime where gravity is so strong that nothing—no particles or even electromagnetic radiation such as light—can escape from it. The boundary beyond which no escape is possible is called the event horizon."
}

@st.cache_resource
def initialize_system():
    model_name = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    
    doc_ids = list(DOCUMENTS.keys())
    doc_contents = list(DOCUMENTS.values())

    st.info(f"Encoding {len(doc_ids)} documents using {model_name}...")
    document_embeddings = model.encode(doc_contents)
    
    dimension = document_embeddings.shape[1]

    faiss.normalize_L2(document_embeddings)
    index = faiss.IndexFlatIP(dimension) 
    
    index.add(document_embeddings)
    st.success("FAISS Index created and documents embedded.")
    
    return model, index, doc_ids, doc_contents

try:
    model, index, doc_ids, doc_contents = initialize_system()
except Exception as e:
    st.error(f"Error initializing RAG system components: {e}")
    st.stop()


def llm_generate_answer(question: str, retrieved_chunks: list[str]) -> str:
    """Feeds the retrieved context and question into the Gemini model."""
    
    context = "\n---\n".join(retrieved_chunks)
    
    prompt = f"""
    You are an expert Question Answering system. Your goal is to answer the user's question only based on the provided CONTEXT. 
    Do NOT use external knowledge. If the CONTEXT does not contain the answer, state that clearly.

    CONTEXT:
    ---
    {context}
    ---
    
    QUESTION: {question}

    ANSWER:
    """
    
    try:
        response = gemini_client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1 # Keep it low for factual Q&A
            )
        )
        return response.text
    except Exception as e:
        return f"LLM Generation Error: {e}"
    
def calculate_confidence(similarity_scores: list[float]) -> float:
    """
    Calculates the confidence score based on the mean of the top K similarity scores.
    Scores are Cosine Similarity, which range from -1 (opposite) to 1 (identical).
    We normalize them to a 0-100% scale for easier interpretation.
    """
    if similarity_scores.size == 0:
        return 0.0

    mean_normalized_score = np.mean([(s + 1) / 2 for s in similarity_scores])
    
    return round(mean_normalized_score * 100, 2)


def retrieve_and_generate(query: str, k: int = 3):
    """Performs the full RAG cycle."""
    query_vector = model.encode([query])
    faiss.normalize_L2(query_vector) 
    D, I = index.search(query_vector, k)
    
    retrieved_indices = I[0]
    similarity_scores = D[0]
    retrieved_info = []
    
    for score, idx in zip(similarity_scores, retrieved_indices):
        if idx >= 0: 
            retrieved_info.append({
                "ID": doc_ids[idx],
                "Score": score,
                "Content": doc_contents[idx]
            })

    top_chunks = [info['Content'] for info in retrieved_info]
    
    final_answer = llm_generate_answer(query, top_chunks)
    
    confidence = calculate_confidence(similarity_scores)
    
    return final_answer, retrieved_info, confidence

st.set_page_config(page_title="Mini RAG QA System", layout="wide")
st.title("🤖 Mini RAG QA System with Gemini API")
st.markdown("A small Retrieval-Augmented Generation system using **Sentence Transformers**, **FAISS**, and the **Gemini API**.")
st.markdown("Try asking about **Mars**, **Quantum Computing**, or **AI Safety**.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask a question about AI, space, or energy..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    final_answer, retrieved_info, confidence = retrieve_and_generate(prompt, k=3)

    retrieval_details = "**Retrieved Documents & Scores:**\n\n"

    table_data = []
    for info in retrieved_info:
        display_score = round((info['Score'] + 1) / 2, 4) 
        table_data.append([info['ID'], f"{info['Score']:.4f}", f"{display_score:.4f} (0-1 Normalized)"])
    
    df = pd.DataFrame(table_data, columns=['Document ID/Title', 'Cosine Sim. Score (-1 to 1)', 'Normalized Score (0-1)'])

    bot_response = f"""
    ### ✨ Final Generated Answer
    {final_answer}
    
    ---
    
    ### 📊 Retrieval and Confidence Details
    
    **Confidence Score (Based on average top 3 scores):** **{confidence}%**
    
    **Top 3 Retrieved Document Scores:**
    """
    
    with st.chat_message("assistant"):
        st.markdown(bot_response)
        st.dataframe(df, hide_index=True)
        
        with st.expander("🔍 Click to see the Full Context Used for Generation"):
            for i, info in enumerate(retrieved_info):
                st.markdown(f"**{i+1}. {info['ID']} (Score: {info['Score']:.4f})**")
                st.code(info['Content'], language='text')

    st.session_state.messages.append({"role": "assistant", "content": bot_response + df.to_markdown(index=False)})