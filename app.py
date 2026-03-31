import streamlit as st
import os, json, tempfile
from pathlib import Path
from datetime import datetime
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="PDF Talker 2.0", page_icon="📚", layout="wide")

# ---------------- ULTRA PRO CSS ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
}
.block-container {
    padding-top: 2rem;
}
.glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(14px);
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    margin-bottom: 20px;
    transition: 0.3s;
}
.glass:hover {
    transform: scale(1.01);
    box-shadow: 0 0 25px rgba(0,255,255,0.3);
}
.logo {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(90deg,#00f260,#0575e6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.heading-box {
    border: 2px solid;
    border-image: linear-gradient(90deg,#ff512f,#dd2476) 1;
    padding: 12px;
    border-radius: 12px;
    text-align: center;
    font-weight: bold;
    font-size: 24px;
    color: white;
    margin-bottom: 15px;
}
.stButton>button {
    background: linear-gradient(90deg,#ff512f,#dd2476);
    border: none;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.08);
    box-shadow: 0 0 20px #ff512f;
}
.stTextInput input {
    border-radius: 10px;
    border: 1px solid #00f260;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#141E30,#243B55);
}
[data-testid="stChatMessage"] {
    border-radius: 15px;
    padding: 10px;
    background: rgba(255,255,255,0.05);
}
.footer {
    text-align: center;
    color: gray;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="logo">S.S_AI 🚀</div>', unsafe_allow_html=True)
st.markdown('<div class="heading-box">📚 PDF Talker 2.0 - Ultra AI System</div>', unsafe_allow_html=True)

st.caption("Powered by Groq LLaMA 3.3 70B · Ultra-Fast Streaming · Professional Urdu/English Support")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "memory" not in st.session_state:
    st.session_state.memory = None

# ---------------- EMBEDDINGS ----------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

# ---------------- CHAIN ----------------
def build_chain(vector_store):
    SYSTEM_PROMPT = """You are a highly intelligent, professional PDF assistant with expert-level understanding.

**CRITICAL RULES:**
1. Answer STRICTLY from the uploaded PDF content only
2. Maintain perfect language consistency - if user writes in Urdu, respond in professional Urdu; if English, respond in professional English
3. NEVER mix Urdu and Hindi - use pure Urdu (اردو) or pure English
4. Provide detailed, comprehensive answers with proper structure
5. Always cite source documents with page numbers
6. If information is not in PDF, politely say: "معذرت، یہ معلومات PDF میں موجود نہیں ہے" (Urdu) or "Apologies, this information is not available in the PDF" (English)
7. Maintain conversation continuity and context from previous messages

Context from PDF:
{context}"""

    qa_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template("{question}"),
    ])

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=8,
    )
    st.session_state.memory = memory

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.25,
        max_tokens=4096,
        api_key=GROQ_API_KEY,
    )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 25, "lambda_mult": 0.5},
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
        verbose=False,
    )

# ---------------- SUMMARY ----------------
def generate_summary(vector_store):
    docs = vector_store.similarity_search("main topics key points overview summary", k=5)
    context = "\n\n".join([d.page_content for d in docs])

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=GROQ_API_KEY
    )

    prompt = f"""Generate a comprehensive 4-5 line summary of this PDF document.
    Language: If content is primarily Urdu, respond in professional Urdu; if English, respond in English.

    Content:
    {context}"""

    response = llm.invoke(prompt)
    return response.content

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

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
                    chunk_size=1200,
                    chunk_overlap=250,
                    separators=["\n\n", "\n", "۔", ".", "،", ",", " ", ""],
                )
                chunks = splitter.split_documents(all_pages)

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
            export_data = [{"role": m["role"], "message": m["content"]} for m in st.session_state.messages]
            filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
            st.download_button("⬇️ Download JSON",
                data=json.dumps(export_data, ensure_ascii=False, indent=2),
                file_name=filename,
                mime="application/json",
                use_container_width=True,
            )

    st.divider()
    st.caption("🚀 Enhanced Features")
    st.caption("• Streaming AI")
    st.caption("• Urdu/English")
    st.caption("• Context Memory")
    st.caption("• LLaMA 3.3 70B")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- MAIN ----------------
if not st.session_state.chain:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.info("👈 Sidebar se PDF upload karo phir 'Process PDFs' dabao")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(f'<div class="glass">{msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("sources"):
            with st.expander("📖 Sources"):
                for src in msg["sources"]:
                    st.caption(f"📄 {src}")

# Chat Input
if user_input := st.chat_input("اپنا سوال پوچھیں... (Urdu ya English)"):
    with st.chat_message("user"):
        st.markdown(f'<div class="glass">{user_input}</div>', unsafe_allow_html=True)

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing PDF..."):
            result = st.session_state.chain.invoke({"question": user_input})
            answer = result["answer"]

            placeholder = st.empty()
            full = ""
            for word in answer.split():
                full += word + " "
                placeholder.markdown(f'<div class="glass">{full}▌</div>', unsafe_allow_html=True)
                time.sleep(0.015)
            placeholder.markdown(f'<div class="glass">{full}</div>', unsafe_allow_html=True)

            sources = []
            seen = set()
            for doc in result["source_documents"]:
                src = f"{doc.metadata.get('file_name','?')} · Page {doc.metadata.get('page',0)+1}"
                if src not in seen:
                    sources.append(src)
                    seen.add(src)

            if sources:
                with st.expander("📖 Source Documents"):
                    for src in sources:
                        st.caption(f"📄 {src}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })

    st.rerun()

# Footer
st.markdown('<div class="footer">© 2026 S.S_AI | Ultra Premium UI</div>', unsafe_allow_html=True)
