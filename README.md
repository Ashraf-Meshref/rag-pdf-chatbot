# 📄 Chat with your PDF (Gemini RAG)

A **Retrieval-Augmented Generation (RAG)** chatbot built with **Streamlit** and **Google Gemini AI**. Upload a PDF and start asking questions — the app retrieves the most relevant chunks from your document and answers using Gemini's generative model.

---

## ✨ Features

- **PDF Upload** – Drag & drop or browse to upload any PDF file.
- **Semantic Search** – Embeddings are computed via `gemini-embedding-001` for accurate retrieval.
- **RAG-based Q&A** – The top-4 most relevant chunks are fed as context to Gemini for grounded answers.
- **Chat History** – Full conversation is preserved during the session.
- **Clean UI** – Built with Streamlit for a simple, responsive interface.

---

## 🛠️ Tech Stack

| Component       | Technology                         |
|-----------------|------------------------------------|
| Frontend        | [Streamlit](https://streamlit.io/) |
| Embeddings      | `gemini-embedding-001` (Google)    |
| LLM             | `gemini-3.5-flash` (Google)        |
| PDF Parsing     | `pypdf`                            |
| Vector Search   | Cosine similarity via NumPy        |
| Language        | Python 3.10+                       |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/RAG_PDF_bot.git
cd RAG_PDF_bot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your Gemini API key

Create a `.streamlit/secrets.toml` file in the project root:

```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
```

> Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).

### 4. Run the app

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
RAG_PDF_bot/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── .streamlit/
    └── secrets.toml        # API key (not committed)
```

---

## ⚙️ How It Works

1. **Upload** a PDF via the sidebar.
2. The app **extracts text** from every page using `pypdf`.
3. Text is **split into chunks** (default 1200 characters each).
4. Each chunk is **embedded** with `gemini-embedding-001` and stored in memory.
5. When you ask a question:
   - The question is embedded with the same model.
   - **Cosine similarity** finds the top-4 most relevant chunks.
   - Those chunks are injected into a prompt as context.
   - Gemini generates an answer **grounded only in that context**.
6. The conversation is displayed in a chat-like interface.

---

## 📦 Dependencies

See [requirements.txt](requirements.txt) for the full list.

Main packages:
- `streamlit` – Web UI framework
- `google-genai` – Google Gemini API client
- `pypdf` – PDF text extraction
- `numpy` – Vector operations & similarity computation

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the MIT License.
