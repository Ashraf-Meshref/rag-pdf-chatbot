import streamlit as st
import numpy as np
from google import genai
from google.genai import types
from pypdf import PdfReader

st.set_page_config(page_title="Chat with your PDF")
st.title("📄 Chat with your PDF (Gemini RAG)")

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
CHAT_MODEL = "gemini-3.1-flash-lite"  # fastest current Gemini model


@st.cache_resource
def get_client():
    return genai.Client(api_key=GEMINI_API_KEY)


client = get_client()


def chunk_text(text, size=1200):
    return [text[i:i + size] for i in range(0, len(text), size) if text[i:i + size].strip()]


def embed(client, texts, query=False):
    task = "RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT"
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=types.EmbedContentConfig(task_type=task),
    )
    return np.array([e.values for e in result.embeddings])


st.session_state.setdefault("messages", [])

if "chunks" not in st.session_state:
    pdf_file = st.file_uploader("Upload a PDF to start chatting with it", type="pdf")

    if pdf_file:
        reader = PdfReader(pdf_file)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        chunks = chunk_text(text)
        with st.spinner("Reading and indexing PDF..."):
            st.session_state.chunks = chunks
            st.session_state.embeddings = embed(client, chunks)
        st.rerun()

else:
    st.caption(f"📚 {len(st.session_state.chunks)} chunks indexed. ")
    if st.button("Upload a different PDF"):
        for key in ("chunks", "embeddings", "messages"):
            st.session_state.pop(key, None)
        st.rerun()

    for m in st.session_state.messages:
        st.chat_message(m["role"]).write(m["content"])

    question = st.chat_input("Ask something about the PDF...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        q_emb = embed(client, [question], query=True)[0]
        embs = st.session_state.embeddings
        sims = embs @ q_emb / (np.linalg.norm(embs, axis=1) * np.linalg.norm(q_emb))
        top_idx = sims.argsort()[-4:][::-1]
        context = "\n---\n".join(st.session_state.chunks[i] for i in top_idx)

        prompt = (
            "Answer the question using only the context below from the PDF. "
            "If the answer isn't in the context, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )

        with st.chat_message("assistant"):
            stream = client.models.generate_content_stream(
                model=CHAT_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="minimal")
                ),
            )
            answer = st.write_stream(chunk.text for chunk in stream if chunk.text)

        st.session_state.messages.append({"role": "assistant", "content": answer})
