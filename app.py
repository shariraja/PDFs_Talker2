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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from groq import RateLimitError

# ==================== RETRY DECORATOR FOR API CALLS ====================
def retry_on_rate_limit(func):
    """Decorator to retry on rate limit errors"""
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RateLimitError)
    )
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

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
    
    /* Animated Border for Sidebar Cards */
    .glass-card {
        background: rgba(15, 15, 26, 0.6);
        backdrop-filter: blur(10px);
        border: 2px solid transparent;
        border-radius: 20px;
        padding: 20px;
        margin: 15px 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        animation: borderPulse 3s infinite;
    }
    
    @keyframes borderPulse {
        0%, 100% {
            border-image: linear-gradient(45deg, #0CF2B3, #4BD9FF, #FF33AA, #0CF2B3) 1;
            border-image-slice: 1;
            box-shadow: 0 0 10px rgba(12, 242, 179, 0.3);
        }
        50% {
            border-image: linear-gradient(225deg, #FF33AA, #4BD9FF, #0CF2B3, #FF33AA) 1;
            border-image-slice: 1;
            box-shadow: 0 0 30px rgba(255, 51, 170, 0.5);
        }
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #0CF2B3, #4BD9FF, #FF33AA, #0CF2B3);
        border-radius: 20px;
        z-index: -1;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .glass-card:hover::before {
        opacity: 1;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 0 40px rgba(12, 242, 179, 0.6);
    }
    
    /* Logo Styling - Animated Box */
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 15px;
        padding: 20px;
        margin-bottom: 30px;
        background: linear-gradient(135deg, rgba(12, 242, 179, 0.15), rgba(255, 51, 170, 0.15));
        border-radius: 15px;
        animation: logoGlow 2s infinite, borderRotate 4s linear infinite;
        position: relative;
        overflow: hidden;
        border: 2px solid transparent;
    }
    
    @keyframes logoGlow {
        0%, 100% { 
            box-shadow: 0 0 20px rgba(12, 242, 179, 0.4);
            border-image: linear-gradient(45deg, #0CF2B3, #FF33AA) 1;
            border-image-slice: 1;
        }
        50% { 
            box-shadow: 0 0 50px rgba(255, 51, 170, 0.6);
            border-image: linear-gradient(225deg, #FF33AA, #0CF2B3) 1;
            border-image-slice: 1;
        }
    }
    
    @keyframes borderRotate {
        0% { border-image: linear-gradient(0deg, #0CF2B3, #FF33AA) 1; }
        100% { border-image: linear-gradient(360deg, #0CF2B3, #FF33AA) 1; }
    }
    
    .logo-text {
        font-size: 36px;
        font-weight: 900;
        background: linear-gradient(135deg, #0CF2B3, #4BD9FF, #FF33AA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'Orbitron', monospace;
        text-shadow: 0 0 20px rgba(12, 242, 179, 0.5);
        letter-spacing: 2px;
        animation: textPulse 2s infinite;
    }
    
    @keyframes textPulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.9; text-shadow: 0 0 30px #FF33AA; }
    }
    
    .logo-icon {
        font-size: 52px;
        animation: iconSpin 3s infinite;
    }
    
    @keyframes iconSpin {
        0%, 100% { 
            transform: rotate(0deg);
            text-shadow: 0 0 10px #0CF2B3;
        }
        50% { 
            transform: rotate(5deg);
            text-shadow: 0 0 30px #FF33AA;
        }
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
        animation: headingGlow 3s infinite;
    }
    
    @keyframes headingGlow {
        0%, 100% { box-shadow: 0 0 20px rgba(12, 242, 179, 0.3); }
        50% { box-shadow: 0 0 40px rgba(255, 51, 170, 0.5); }
    }
    
    .neon-heading:hover {
        transform: scale(1.02);
        box-shadow: 0 0 50px rgba(12, 242, 179, 0.5);
        border-color: #FF33AA;
    }
    
    /* Sidebar text color */
    [data-testid="stSidebar"] * {
        color: #0CF2B3 !important;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4 {
        color: #FF33AA !important;
        text-shadow: 0 0 5px rgba(255, 51, 170, 0.5);
    }
    
    /* WHITE TEXT FOR CHAT */
    .user-message {
        background: linear-gradient(135deg, rgba(12, 242, 179, 0.15), rgba(75, 217, 255, 0.1));
        border-left: 4px solid #0CF2B3;
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        animation: slideInRight 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        color: #FFFFFF !important;
        font-family: 'Share Tech Mono', monospace;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .user-message:hover {
        transform: translateX(5px);
        border-left: 4px solid #FF33AA;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, rgba(255, 51, 170, 0.15), rgba(12, 242, 179, 0.1));
        border-right: 4px solid #FF33AA;
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        animation: slideInLeft 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        color: #FFFFFF !important;
        font-family: 'Share Tech Mono', monospace;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .assistant-message:hover {
        transform: translateX(-5px);
        border-right: 4px solid #0CF2B3;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(100px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-100px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Chat input */
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input {
        color: #FFFFFF !important;
        font-family: 'Share Tech Mono', monospace !important;
    }
    
    [data-testid="stChatInput"] > div {
        background: rgba(15, 15, 26, 0.8);
        border: 2px solid rgba(12, 242, 179, 0.5);
        border-radius: 15px;
        transition: all 0.3s ease;
        animation: inputGlow 2s infinite;
    }
    
    @keyframes inputGlow {
        0%, 100% { border-color: rgba(12, 242, 179, 0.5); }
        50% { border-color: rgba(255, 51, 170, 0.5); }
    }
    
    [data-testid="stChatInput"] > div:focus-within {
        border-color: #FF33AA;
        box-shadow: 0 0 30px rgba(255, 51, 170, 0.5);
    }
    
    /* Streaming cursor */
    .streaming-cursor {
        display: inline-block;
        width: 3px;
        height: 20px;
        background: #FFFFFF;
        animation: blink 1s infinite;
        margin-left: 5px;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
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
</style>

<script>
function scrollToBottom() {
    const mainElement = document.querySelector('.main');
    if (mainElement) {
        mainElement.scrollTo({
            top: mainElement.scrollHeight,
            behavior: 'smooth'
        });
    }
}

const observer = new MutationObserver(function(mutations) {
    scrollToBottom();
});

const mainElement = document.querySelector('.main');
if (mainElement) {
    observer.observe(mainElement, { childList: true, subtree: true });
    scrollToBottom();
}
</script>
""", unsafe_allow_html=True)

# ==================== LOGO SECTION ====================
st.markdown("""
<div class="logo-container">
    <div class="logo-icon">⚡</div>
    <div class="logo-text">S.S_AI</div>
    <div style="font-size: 12px; color: #0CF2B3; margin-left: 10px;">NEURAL PDF ASSISTANT v2.0</div>
</div>
""", unsafe_allow_html=True)

# ==================== BACKEND CODE ====================
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "memory" not in st.session_state:
    st.session_state.memory = None
if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = 0

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

@retry_on_rate_limit
def invoke_with_retry(llm, prompt):
    """Wrapper function with retry logic for API calls"""
    return llm.invoke(prompt)

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
        request_timeout=60,
        max_retries=3,
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
        api_key=GROQ_API_KEY,
        request_timeout=60,
        max_retries=3,
    )
    
    prompt = f"""Generate a comprehensive 4-5 line summary of this PDF document.
    Language: If content is primarily Urdu, respond in professional Urdu; if English, respond in English.
    Make it concise but informative, highlighting key points.
    
    Content:
    {context}"""
    
    # Add delay between requests to avoid rate limiting
    time.sleep(1)
    
    try:
        response = invoke_with_retry(llm, prompt)
        return response.content
    except Exception as e:
        st.error(f"⚠️ Rate limit reached. Please wait a moment and try again.")
        return "Summary temporarily unavailable. Please try again in a few seconds."

# ==================== MAIN APP LAYOUT ====================
st.markdown('<h1 class="neon-heading">📚 NEURAL PDF INTERFACE</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #0CF2B3; font-family: monospace;">⚡ POWERED BY GROQ LLaMA 3.3 70B | ULTRA-FAST STREAMING ⚡</p>', unsafe_allow_html=True)

# Sidebar
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

# Display Chat History
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

# Chat Input with Rate Limit Handling
if user_input := st.chat_input("⚡ ENTER YOUR QUERY... (URDU | ENGLISH)"):
    with st.chat_message("user"):
        st.markdown(f'<div class="user-message">{user_input}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        # Check rate limit
        current_time = time.time()
        if current_time - st.session_state.last_request_time < 2:
            st.warning("⚠️ Please wait a moment between messages (rate limit)")
            time.sleep(2)
        
        with st.spinner("🧠 NEURAL PROCESSING..."):
            try:
                result = st.session_state.chain.invoke({"question": user_input})
                answer = result["answer"]
                st.session_state.last_request_time = time.time()
                
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
                            
            except Exception as e:
                st.error("⚠️ Rate limit reached. Please wait 10-15 seconds before asking another question.")
                answer = "⚠️ Too many requests. Please wait a moment and try again."
                st.markdown(f'<div class="assistant-message">{answer}</div>', unsafe_allow_html=True)
                sources = []

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
    
    st.rerun()
