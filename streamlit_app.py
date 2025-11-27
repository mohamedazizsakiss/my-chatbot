import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Smart Store AI", page_icon="🤖")

@st.cache_resource
def load_models():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_models()

# --- 2. DATA (Your Knowledge Base) ---
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = [
        {"id": 1, "content": "We ship all domestic orders via FedEx. Delivery takes 3-5 business days."},
        {"id": 2, "content": "Returns are accepted within 30 days of purchase if they are unused."},
        {"id": 3, "content": "We accept Visa, Mastercard, PayPal, and Apple Pay."},
        {"id": 4, "content": "Support email: help@mystore.com. Phone: 1-800-555-0199."},
        {"id": 5, "content": "Our store is open Mon-Fri from 9 AM to 5 PM EST."}
    ]

# --- 3. VECTORIZATION (Retrieval) ---
def get_best_match(query):
    corpus = [d['content'] for d in st.session_state.knowledge_base]
    corpus_embeddings = embedding_model.encode(corpus)
    query_embedding = embedding_model.encode([query])
    
    scores = cosine_similarity(query_embedding, corpus_embeddings)[0]
    best_idx = np.argmax(scores)
    return corpus[best_idx], scores[best_idx]

# --- 4. GENERATION (The Brain) ---
def generate_answer(query):
    try:
        # Use the Secret Key from Streamlit Cloud
        my_key = st.secrets["GEMINI_KEY"]
        
        genai.configure(api_key=my_key)
        model = genai.GenerativeModel('models/gemini-1.5-flash')

        # A. Retrieve Context
        best_text, score = get_best_match(query)
        
        # B. Build History (Memory of last 5 messages)
        history_text = ""
        for msg in st.session_state.messages[-5:]:
            history_text += f"{msg['role']}: {msg['content']}\n"

        # C. Decision Logic (Smart Router)
        if score > 0.35:
            # High match? Use Database
            prompt = f"""
            You are a helpful store assistant.
            Use the CONTEXT and HISTORY to answer.
            
            CONTEXT: {best_text}
            HISTORY: {history_text}
            USER QUESTION: {query}
            """
        else:
            # Low match? Just Chit-Chat
            prompt = f"""
            You are a friendly customer service AI. 
            The user is saying hello or asking a general question.
            Be polite and helpful. Do NOT make up store info.
            
            HISTORY: {history_text}
            USER SAID: {query}
            """

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Error: {str(e)}"

# --- 5. UI (Chat Window) ---
st.title("🤖 Smart Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask me..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_answer(prompt)
            st.write(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})