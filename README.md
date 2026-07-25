# 🩺 Medical Chatbot — LangChain + Pinecone (Groq / Ollama, no OpenAI)

A Retrieval-Augmented-Generation (RAG) medical chatbot built with:

- **LangChain** for orchestration
- **Pinecone** as the vector database
- **HuggingFace sentence-transformers** for embeddings
- **Groq API** or **Ollama (local)** as the LLM — **OpenAI is not used anywhere**
- Two ready-to-use front ends:
  - `app.py` — the original **Flask** web UI (`templates/chat.html` + `static/style.css`)
  - `streamlit_app.py` — a **Streamlit** chat UI with a sidebar to switch provider/model live

---

## 1. Clone & set up environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Configure environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```
PINECONE_API_KEY=your_pinecone_api_key_here
GROQ_API_KEY=your_groq_api_key_here          # only needed if using Groq
OLLAMA_BASE_URL=http://localhost:11434       # only needed if using Ollama

# Used by the Flask app (app.py) to pick the provider:
LLM_PROVIDER=groq                            # "groq" or "ollama"
# LLM_MODEL=llama-3.3-70b-versatile          # optional override
```

- Get a free Pinecone key: https://app.pinecone.io
- Get a free Groq key: https://console.groq.com/keys

## 3. Index your documents into Pinecone

Put your PDF(s) in the `data/` folder (a sample `Medical_book.pdf` is included), then run:

```bash
python store_index.py
```

This chunks the PDF(s), embeds them with `sentence-transformers/all-MiniLM-L6-v2`, and
upserts them into a Pinecone index called `medical-chatbot`.

## 4. Run the app

### Option A — Streamlit UI (recommended, lets you switch provider/model in the browser)

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501`. In the sidebar you can:
- Switch between **Groq** (cloud) and **Ollama** (local) as the LLM provider
- Pick a specific model for the chosen provider
- Paste/edit your Groq API key directly in the UI (or leave it to load from `.env`)

### Option B — Flask UI (original chat.html interface)

```bash
python app.py
```

Open `http://localhost:8080`. The provider/model are set via `LLM_PROVIDER` /
`LLM_MODEL` in your `.env` file (defaults to Groq, model `llama-3.3-70b-versatile`).

### Using Ollama locally

1. Install Ollama: https://ollama.com
2. Pull a model, e.g.:
   ```bash
   ollama pull llama3.1
   ```
3. Make sure the Ollama server is running (`ollama serve`, or it auto-starts on install)
4. Streamlit: select **Ollama** in the sidebar. Flask: set `LLM_PROVIDER=ollama` in `.env`.

### Using Groq

1. Create a free API key at https://console.groq.com/keys
2. Paste it into the Streamlit sidebar, or put it in `.env` as `GROQ_API_KEY`
3. Pick a model (e.g. `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`)

---

## Project structure

```
├── .github/workflows/cicd.yaml   # CI/CD to build & deploy the Docker image (AWS ECR/EC2)
├── data/                         # source PDFs to index
├── research/trials.ipynb         # exploratory notebook
├── src/
│   ├── helper.py                 # PDF loading, chunking, embeddings
│   ├── prompt.py                 # system prompt for the RAG chain
│   └── llm.py                    # Groq / Ollama LLM factory (no OpenAI)
├── static/style.css              # styling for the Flask chat UI
├── templates/chat.html           # Flask chat UI
├── app.py                        # Flask entry point
├── streamlit_app.py              # Streamlit entry point
├── store_index.py                # one-off script to embed + upsert docs into Pinecone
├── setup.py
├── template.sh                   # scaffolding script for a fresh project
├── requirements.txt
├── Dockerfile                    # builds & runs the Flask app (port 8080)
├── Dockerfile.streamlit          # builds & runs the Streamlit app (port 8501)
└── .env.example
```

## Docker

Flask (matches the CI/CD pipeline, port 8080):

```bash
docker build -t medical-chatbot .
docker run -p 8080:8080 --env-file .env medical-chatbot
```

Streamlit (port 8501):

```bash
docker build -f Dockerfile.streamlit -t medical-chatbot-streamlit .
docker run -p 8501:8501 --env-file .env medical-chatbot-streamlit
```

> Note: if you use Ollama with Docker, you'll need Ollama reachable from inside the
> container (e.g. run Ollama on the host and set `OLLAMA_BASE_URL=http://host.docker.internal:11434`).

## Notes

- This project deliberately avoids any OpenAI dependency — the only LLM providers wired
  up are **Groq** and **Ollama**.
- Embeddings still run locally via `sentence-transformers` (no API key required for that part).
- The GitHub Actions workflow (`.github/workflows/cicd.yaml`) passes `PINECONE_API_KEY` and
  `GROQ_API_KEY` as secrets to the deployed container instead of `OPENAI_API_KEY`.
