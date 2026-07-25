import os
import streamlit as st
from dotenv import load_dotenv

from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt
from src.llm import get_llm, GROQ_MODELS, OLLAMA_MODELS

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------
load_dotenv()

st.set_page_config(page_title="Medical Chatbot", page_icon="🩺", layout="wide")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

INDEX_NAME = "medical-chatbot"


@st.cache_resource(show_spinner="Loading embeddings & vector store...")
def load_retriever():
    embeddings = download_hugging_face_embeddings()
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
    )
    return docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})


def build_rag_chain(provider: str, model: str):
    retriever = load_retriever()
    llm = get_llm(provider=provider, model=model)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain


# --------------------------------------------------------------------------
# Sidebar: provider / model / API key configuration
# --------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

provider = st.sidebar.radio(
    "LLM Provider",
    options=["groq", "ollama"],
    format_func=lambda p: "Groq (cloud, needs API key)" if p == "groq" else "Ollama (local)",
)

if provider == "groq":
    model = st.sidebar.selectbox("Groq model", GROQ_MODELS)

    default_key = os.environ.get("GROQ_API_KEY", "")
    groq_key_input = st.sidebar.text_input(
        "GROQ_API_KEY",
        value=default_key,
        type="password",
        help="Get a free key at https://console.groq.com/keys. "
             "You can also set GROQ_API_KEY in a .env file instead.",
    )
    if groq_key_input:
        os.environ["GROQ_API_KEY"] = groq_key_input

else:
    model = st.sidebar.selectbox("Ollama model", OLLAMA_MODELS)
    ollama_url = st.sidebar.text_input(
        "Ollama base URL",
        value=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Make sure `ollama serve` is running and the model has been pulled, "
             "e.g. `ollama pull llama3.1`",
    )
    os.environ["OLLAMA_BASE_URL"] = ollama_url

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear chat"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown(
    "This app does **not** use OpenAI. It uses either the **Groq API** "
    "or a **local Ollama** model as the LLM, with Pinecone as the vector store."
)

# --------------------------------------------------------------------------
# Main chat UI
# --------------------------------------------------------------------------
st.title("🩺 Medical Chatbot")
st.caption("RAG chatbot powered by LangChain + Pinecone, running on Groq or Ollama.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a medical question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if provider == "groq" and not os.environ.get("GROQ_API_KEY"):
            answer = "⚠️ Please enter your GROQ_API_KEY in the sidebar before chatting."
            st.markdown(answer)
        else:
            try:
                with st.spinner(f"Thinking with {provider}:{model}..."):
                    rag_chain = build_rag_chain(provider, model)
                    response = rag_chain.invoke({"input": user_input})
                    answer = response["answer"]
                st.markdown(answer)
            except Exception as e:
                answer = f"⚠️ Error: {e}"
                st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
