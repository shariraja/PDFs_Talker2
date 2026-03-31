import streamlit as st
import os, json, tempfile
from pathlib import Path
from datetime import datetime
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  # Changed from Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

st.set_page_config(page_title="PDF Talker 2.0", page_icon="📚", layout="wide")
st.title("📚 PDF Talker 2.0")
st.caption("Powered by Groq LLaMA 3.3 · Free · Fast · Urdu Support")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# ─── Session State ────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "memory" not in st.session_state:
    st.session_state.memory = None

# ─── Embeddings ───────────────────────────────────────────────────
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

# ─── Chain ────────────────────────────────────────────────────────
def build_chain(vector_store):
    SYSTEM_PROMPT = """You are an intelligent PDF assistant.
Answer ONLY from the uploaded PDF content.

Rules:
- If user writes in Urdu, reply in Urdu
- If user writes in English, reply in English
- If answer not in PDF, say: 'Yeh information PDF mein nahi hai'
- Always mention source file and page number

Context:
{context}"""

    qa_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template("{question}"),
    ])

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=5,
    )
    st.session_state.memory = memory

    return ConversationalRetrievalChain.from_llm(
        llm=ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, api_key=GROQ_API_KEY),
        retriever=vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 15, "lambda_mult": 0.7},
        ),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
    )

# ─── Summary ──────────────────────────────────────────────────────
def generate_summary(vector_store):
    docs = vector_store.similarity_search("main topic overview summary", k=3)
    context = "\n\n".join([d.page_content for d in docs])
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, api_key=GROQ_API_KEY)
    response = llm.invoke(
        f"PDF ka 3-4 line ka summary do. Sirf English ya Urdu ya Roman Urdu mein likho, Hindi bilkul mat likho:\n\n{context}"
    )
    return response.content

# ─── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 PDF Upload")
    uploaded_files = st.file_uploader("PDFs select karo", type=["pdf"], accept_multiple_files=True)

    if st.button("⚡ Process PDFs", use_container_width=True):
        if not uploaded_files:
            st.warning("Pehle PDF select karo!")
        else:
            with st.spinner("PDFs process ho rahi hain..."):
                all_pages = []
                for f in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.read())
                        tmp_path = tmp.name
                    loader = PyMuPDFLoader(tmp_path)
                    pages = loader.load()
                    for page in pages:
                        page.metadata["file_name"] = f.name
                    all_pages.extend(pages)
                    os.unlink(tmp_path)

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200,
                    separators=["\n\n", "\n", ".", " "],
                )
                chunks = splitter.split_documents(all_pages)

                # Changed: Using FAISS instead of Chroma
                vector_store = FAISS.from_documents(
                    documents=chunks,
                    embedding=get_embeddings()
                )

                st.session_state.chain = build_chain(vector_store)
                st.session_state.messages = []

            with st.spinner("Summary generate ho rahi hai..."):
                st.session_state.summary = generate_summary(vector_store)

            st.success(f"✅ {len(uploaded_files)} PDF(s) ready!")

    if st.session_state.summary:
        st.divider()
        st.subheader("📋 PDF Summary")
        st.info(st.session_state.summary)

    st.divider()
    if st.session_state.messages:
        if st.button("💾 Export Chat", use_container_width=True):
            export_data = [
                {"role": m["role"], "message": m["content"]}
                for m in st.session_state.messages
            ]
            filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
            st.download_button(
                label="⬇️ Download JSON",
                data=json.dumps(export_data, ensure_ascii=False, indent=2),
                file_name=filename,
                mime="application/json",
                use_container_width=True,
            )

# ─── Chat Area ────────────────────────────────────────────────────
if not st.session_state.chain:
    st.info("👈 Sidebar se PDF upload karo phir 'Process PDFs' dabao")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📖 Sources"):
                for src in msg["sources"]:
                    st.caption(f"📄 {src}")

if user_input := st.chat_input("Apna sawaal poochho... (Urdu ya English)"):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Soch raha hoon..."):
            result = st.session_state.chain.invoke({"question": user_input})
            answer = result["answer"]
            sources = []
            seen = set()
            for doc in result["source_documents"]:
                src = f"{doc.metadata.get('file_name', '?')} · Page {doc.metadata.get('page', 0) + 1}"
                if src not in seen:
                    sources.append(src)
                    seen.add(src)
        st.markdown(answer)
        if sources:
            with st.expander("📖 Sources"):
                for src in sources:
                    st.caption(f"📄 {src}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
