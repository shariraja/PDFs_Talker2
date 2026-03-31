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

# ==================== CUSTOM CYBERPUNK CSS ====================
st.set_page_config(
    page_title="S.S_AI | Neural PDF Assistant", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Cyberpunk/Futuristic Theme
st.markdown("""
<style>
    /* Import Cyberpunk Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 50%, #0a0a0f 100%);
        font-family: 'Orbitron', monospace;
    }
    
    /* Glassmorphism Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(10, 10, 20, 0.75) !important;
        backdrop-filter: blur(20px);
        border-right: 2px solid rgba(12, 242, 179, 0.3);
        box-shadow: -5px 0 30px rgba(12, 242, 179, 0.1);
    }
    
    /* Logo Styling */
    .logo-container {
        display: flex;
        align-items: center;
        gap: 15px;
        padding: 20px;
        margin-bottom: 30px;
        border-bottom: 2px solid rgba(12, 242, 179, 0.3);
    }
    
    .logo-text {
        font-size: 32px;
        font-weight: 900;
        background: linear-gradient(135deg, #0CF2B3, #4BD9FF, #FF33AA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'Orbitron', monospace;
        text-shadow: 0 0 20px rgba(12, 242, 179, 0.5);
    }
    
    .logo-icon {
        font-size: 48px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { text-shadow: 0 0 10px #0CF2B3; }
        50% { text-shadow: 0 0 30px #FF33AA; }
    }
    
    /* Neon Gradient Headings */
    .neon-heading {
        background: linear-gradient(135deg, #0CF2B3, #4BD9FF, #FF33AA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900;
        text-align: center;
        padding: 15px;
        border: 2px solid rgba(12, 242, 179, 0.5);
        border-radius: 15px;
        box-shadow: 0 0 30px rgba(12, 242, 179, 0.2);
        transition: all 0.3s ease;
    }
    
    .neon-heading:hover {
        transform: scale(1.02);
        box-shadow: 0 0 50px rgba(12, 242, 179, 0.5);
        border-color: #FF33AA;
    }
    
    /* Glassmorphism Card */
    .glass-card {
        background: rgba(15, 15, 26, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(12, 242, 179, 0.3);
        border-radius: 20px;
        padding: 20px;
        margin: 15px 0;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: #FF33AA;
        box-shadow: 0 0 20px rgba(255, 51, 170, 0.2);
        transform: translateY(-2px);
    }
    
    /* Neon Button */
    .stButton > button {
        background: linear-gradient(135deg, rgba(12, 242, 179, 0.2), rgba(255, 51, 170, 0.2));
        border: 2px solid #0CF2B3;
        color: #0CF2B3;
        font-weight: bold;
        font-family: 'Orbitron', monospace;
        transition: all 0.3s ease;
        border-radius: 10px;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(12, 242, 179, 0.6);
        border-color: #FF33AA;
        color: #FF33AA;
    }
    
    /* Neon File Uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(12, 242, 179, 0.5);
        border-radius: 15px;
        padding: 20px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #FF33AA;
        box-shadow: 0 0 20px rgba(255, 51, 170, 0.3);
    }
    
    /* Chat Bubbles */
    .user-message {
        background: linear-gradient(135deg, rgba(12, 242, 179, 0.15), rgba(75, 217, 255, 0.1));
        border-left: 4px solid #0CF2B3;
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        animation: slideInRight 0.3s ease;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, rgba(255, 51, 170, 0.15), rgba(12, 242, 179, 0.1));
        border-right: 4px solid #FF33AA;
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        animation: slideInLeft 0.3s ease;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Glowing Text */
    .glow-text {
        color: #0CF2B3;
        text-shadow: 0 0 10px rgba(12, 242, 179, 0.5);
        font-family: 'Share Tech Mono', monospace;
    }
    
    /* Neon Cursor for Streaming */
    .streaming-cursor {
        display: inline-block;
        width: 3px;
        height: 20px;
        background: #0CF2B3;
        animation: blink 1s infinite;
        margin-left: 5px;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    
    /* Neon Border Expandable */
    .streamlit-expanderHeader {
        background: rgba(12, 242, 179, 0.1);
        border: 1px solid #0CF2B3;
        border-radius: 10px;
        font-family: 'Orbitron', monospace;
        color: #0CF2B3;
    }
    
    .streamlit-expanderContent {
        background: rgba(15, 15, 26, 0.6);
        border-left: 2px solid #FF33AA;
    }
    
    /* Chat Input */
    [data-testid="stChatInput"] > div {
        background: rgba(15, 15, 26, 0.8);
        border: 2px solid rgba(12, 242, 179, 0.5);
        border-radius: 15px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stChatInput"] > div:focus-within {
        border-color: #FF33AA;
        box-shadow: 0 0 20px rgba(255, 51, 170, 0.3);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #0CF2B3 transparent #FF33AA transparent;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a0a0f;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #0CF2B3, #FF33AA);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #FF33AA;
    }
    
    /* Info/Warning/Success Boxes */
    .stAlert {
        background: rgba(12, 242, 179, 0.1);
        border: 1px solid #0CF2B3;
        border-radius: 10px;
        font-family: 'Share Tech Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOGO SECTION ====================
st.markdown("""
<div class="logo-container">
    <div class="logo-icon">⚡</div>
    <div class="logo-text">S.S_AI</div>
    <div style="margin-left: auto; font-size: 12px; color: #0CF2B3;">NEURAL PDF ASSISTANT v2.0</div>
</div>
""", unsafe_allow_html=True)

# ==================== BACKEND CODE (UNCHANGED) ====================
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "memory" not in st.session_state:
    st.session_state.memory = None

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

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

**Response Structure:**
- Main answer with clear explanation
- Supporting details from PDF
- Source references

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
        search_kwargs={
            "k": 8,
            "fetch_k": 25,
            "lambda_mult": 0.5,
        },
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
        verbose=False,
    )

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
    Make it concise but informative, highlighting key points.
    
    Content:
    {context}"""
    
    response = llm.invoke(prompt)
    return response.content

def stream_response(answer):
    placeholder = st.empty()
    full_response = ""
    words = answer.split()
    for i, word in enumerate(words):
        full_response += word + " "
        placeholder.markdown(f"{full_response}<span class='streaming-cursor'></span>", unsafe_allow_html=True)
        time.sleep(0.02)
    placeholder.markdown(full_response)
    return full_response

# ==================== MAIN APP LAYOUT ====================
# Main Title with Neon Effect
st.markdown('<h1 class="neon-heading">📚 NEURAL PDF INTERFACE</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #0CF2B3; font-family: monospace;">⚡ POWERED BY GROQ LLaMA 3.3 70B | ULTRA-FAST STREAMING ⚡</p>', unsafe_allow_html=True)

# Sidebar with Glassmorphism
with st.sidebar:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #0CF2B3; text-align: center;">📄 UPLOAD INTERFACE</h3>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("SELECT PDF FILES", type=["pdf"], accept_multiple_files=True)

    if st.button("⚡ INITIATE PROCESSING", use_container_width=True):
        if not uploaded_files:
            st.warning("⚠️ NO PDF DETECTED")
        else:
            with st.spinner("🔮 PROCESSING NEURAL NETWORKS..."):
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

            with st.spinner("📊 GENERATING NEURAL SUMMARY..."):
                st.session_state.summary = generate_summary(vector_store)

            st.success(f"✅ {len(uploaded_files)} PDF(S) INTEGRATED")

    if st.session_state.summary:
        st.divider()
        st.markdown('<h3 style="color: #FF33AA; text-align: center;">📋 NEURAL SUMMARY</h3>', unsafe_allow_html=True)
        st.markdown(f'<div class="glass-card" style="border-color: #FF33AA;">{st.session_state.summary}</div>', unsafe_allow_html=True)

    st.divider()
    if st.session_state.messages:
        if st.button("💾 EXPORT NEURAL LOG", use_container_width=True):
            export_data = [
                {"role": m["role"], "message": m["content"]}
                for m in st.session_state.messages
            ]
            filename = f"neural_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
            st.download_button(
                label="⬇️ DOWNLOAD JSON",
                data=json.dumps(export_data, ensure_ascii=False, indent=2),
                file_name=filename,
                mime="application/json",
                use_container_width=True,
            )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card" style="margin-top: 20px; text-align: center;">
        <small style="color: #0CF2B3;">⚡ SYSTEM STATUS: ONLINE</small><br>
        <small style="color: #FF33AA;">🔮 AI CORE: LLaMA 3.3 70B</small><br>
        <small style="color: #4BD9FF;">📡 STREAMING: ACTIVE</small>
    </div>
    """, unsafe_allow_html=True)

# ==================== CHAT INTERFACE ====================
if not st.session_state.chain:
    st.markdown("""
    <div class="glass-card" style="text-align: center; margin: 50px;">
        <h3 style="color: #0CF2B3;">⚡ NEURAL INTERFACE READY</h3>
        <p style="color: #4BD9FF;">UPLOAD PDF DOCUMENTS TO INITIATE NEURAL PROCESSING</p>
        <p style="font-size: 12px; color: #FF33AA;">[ SIDE BAR CONTROLS ACTIVE ]</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Display Chat History with Custom Styling
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f'<div class="user-message">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        with st.chat_message("assistant"):
            st.markdown(f'<div class="assistant-message">{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander("📖 SOURCE DOCUMENTS"):
                    for src in msg["sources"]:
                        st.caption(f"📄 {src}")

# Chat Input with Streaming
if user_input := st.chat_input("⚡ ENTER YOUR QUERY... (URDU | ENGLISH)"):
    with st.chat_message("user"):
        st.markdown(f'<div class="user-message">{user_input}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("🧠 NEURAL PROCESSING..."):
            result = st.session_state.chain.invoke({"question": user_input})
            answer = result["answer"]
            
            response_placeholder = st.empty()
            full_response = ""
            words = answer.split()
            for word in words:
                full_response += word + " "
                response_placeholder.markdown(f'<div class="assistant-message">{full_response}<span class="streaming-cursor"></span></div>', unsafe_allow_html=True)
                time.sleep(0.015)
            response_placeholder.markdown(f'<div class="assistant-message">{full_response}</div>', unsafe_allow_html=True)
            
            sources = []
            seen = set()
            for doc in result["source_documents"]:
                src = f"{doc.metadata.get('file_name', '?')} · PAGE {doc.metadata.get('page', 0) + 1}"
                if src not in seen:
                    sources.append(src)
                    seen.add(src)
            
            if sources:
                with st.expander("📖 SOURCE DOCUMENTS"):
                    for src in sources:
                        st.caption(f"📄 {src}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
    
    st.rerun()
