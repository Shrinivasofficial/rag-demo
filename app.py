import streamlit as st
import google.generativeai as genai
import faiss
import numpy as np
from pypdf import PdfReader
import tempfile
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Gemini RAG Assistant", layout="wide")
st.title("Context-Aware AI Assistant")

# -----------------------------
# SESSION STATE INIT
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_index" not in st.session_state:
    st.session_state.vector_index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "ready" not in st.session_state:
    st.session_state.ready = False

# -----------------------------
# API KEY INPUT
# -----------------------------
api_key = st.text_input("Enter Gemini API Key", type="password")

if api_key:
    genai.configure(api_key=api_key)

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload a PDF Document", type="pdf")

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

def chunk_text(text, chunk_size=800):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def embed_texts(text_list):
    embeddings = []
    for text in text_list:
        response = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text
        )
        embeddings.append(response["embedding"])
    return embeddings

# -----------------------------
# INGESTION BLOCK (RUNS ONCE)
# -----------------------------
if uploaded_file and api_key and not st.session_state.ready:

    with st.spinner("Ingesting document and building knowledge index..."):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        full_text = extract_text_from_pdf(tmp_path)
        chunks = chunk_text(full_text)

        embeddings = embed_texts(chunks)

        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype("float32"))

        st.session_state.vector_index = index
        st.session_state.chunks = chunks
        st.session_state.ready = True

        os.remove(tmp_path)

    st.success("Document indexed successfully. You can now start chatting.")

# -----------------------------
# CHAT BLOCK (INDEPENDENT)
# -----------------------------
if st.session_state.ready:

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask something about the document...")

    if user_input:

        # Exit condition
        if user_input.lower().strip() in ["end", "thanks", "thank you"]:
            st.success("Session ended. Refresh to start again.")
            st.stop()

        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # -----------------------------
        # RETRIEVAL
        # -----------------------------
        query_embedding = genai.embed_content(
            model="models/gemini-embedding-001",
            content=user_input
        )["embedding"]

        D, I = st.session_state.vector_index.search(
            np.array([query_embedding]).astype("float32"),
            k=3
        )

        retrieved_chunks = [st.session_state.chunks[i] for i in I[0]]
        context = "\n\n".join(retrieved_chunks)

        # -----------------------------
        # MEMORY LAYER (LAST 6 MESSAGES)
        # -----------------------------
        conversation_history = ""
        for msg in st.session_state.messages[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_history += f"{role}: {msg['content']}\n"

        # -----------------------------
        # STRONG SYSTEM PROMPT
        # -----------------------------
        system_prompt = f"""
You are an expert AI assistant operating in strict retrieval mode.

Core Rules:
1. Use ONLY the provided context.
2. If the answer is not clearly present in the context, respond:
   "The document does not provide sufficient information."
3. Do not hallucinate.
4. Be concise, structured, and professional.
5. Maintain conversation continuity using previous exchanges.

Retrieved Context:
{context}

Conversation History:
{conversation_history}

Current Question:
{user_input}
"""

        # -----------------------------
        # GENERATION
        # -----------------------------
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(system_prompt)

        assistant_reply = response.text

        # Show assistant reply
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)

        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_reply}
        )

# -----------------------------
# OPTIONAL RESET BUTTON
# -----------------------------
if st.session_state.ready:
    if st.button("Reset Session"):
        st.session_state.messages = []
        st.session_state.ready = False
        st.session_state.vector_index = None
        st.session_state.chunks = []
        st.success("Session reset.")
