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
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RateLimitError)
    )
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="S.S_AI | Neural PDF Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== MASTER CSS ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');

/* ─── ROOT VARIABLES ─── */
:root {
    --bg-base:       #06090f;
    --bg-mid:        #0c1220;
    --bg-card:       #111927;
    --bg-hover:      #162035;
    --cyan:          #0CF2B3;
    --cyan-dim:      rgba(12, 242, 179, 0.18);
    --cyan-glow:     rgba(12, 242, 179, 0.55);
    --pink:          #FF3FA4;
    --pink-dim:      rgba(255, 63, 164, 0.18);
    --pink-glow:     rgba(255, 63, 164, 0.55);
    --blue:          #4BD9FF;
    --blue-dim:      rgba(75, 217, 255, 0.18);
    --gold:          #F4C430;
    --gold-dim:      rgba(244, 196, 48, 0.18);
    --white:         #F0F6FF;
    --white-soft:    #C8D8EE;
    --grey:          #8A9BB8;
    --border-subtle: rgba(75, 217, 255, 0.15);
    --border-active: rgba(12, 242, 179, 0.5);
    --shadow-card:   0 8px 32px rgba(0,0,0,0.5);
    --radius:        14px;
    --radius-sm:     8px;
}

/* ─── GLOBAL RESET ─── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background: var(--bg-base) !important;
    font-family: 'Exo 2', sans-serif;
    color: var(--white) !important;
}

/* ─── SCROLLBAR ─── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--cyan), var(--pink));
    border-radius: 3px;
}

/* ─── SIDEBAR ─── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080e1a 0%, #0b1220 100%) !important;
    border-right: 1px solid var(--border-subtle);
}
[data-testid="stSidebar"] > div { padding-top: 10px; }

/* Force all sidebar text white/readable */
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color: var(--white-soft) !important; }

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 { color: var(--cyan) !important; }

/* ─── LOGO BOX ─── */
.ss-logo-wrap {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 20px;
    margin: 0 0 8px 0;
    background: linear-gradient(135deg, rgba(12,242,179,0.08), rgba(255,63,164,0.08));
    border: 1px solid rgba(12,242,179,0.25);
    border-radius: var(--radius);
    animation: logoBreath 4s ease-in-out infinite;
    position: relative;
    overflow: hidden;
}
.ss-logo-wrap::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 60%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(12,242,179,0.08), transparent);
    animation: logoShimmer 3.5s ease-in-out infinite;
}
@keyframes logoBreath {
    0%,100% { box-shadow: 0 0 16px rgba(12,242,179,0.2); border-color: rgba(12,242,179,0.25); }
    50%      { box-shadow: 0 0 30px rgba(255,63,164,0.3); border-color: rgba(255,63,164,0.35); }
}
@keyframes logoShimmer {
    0%   { left: -100%; }
    100% { left: 200%; }
}
.ss-logo-icon {
    font-size: 38px;
    line-height: 1;
    animation: iconFloat 3s ease-in-out infinite;
    filter: drop-shadow(0 0 8px var(--cyan));
}
@keyframes iconFloat {
    0%,100% { transform: translateY(0px) rotate(0deg); }
    50%      { transform: translateY(-4px) rotate(5deg); }
}
.ss-logo-text {
    font-family: 'Orbitron', monospace;
    font-size: 28px;
    font-weight: 900;
    background: linear-gradient(120deg, var(--cyan), var(--blue), var(--pink));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 3px;
}
.ss-logo-tag {
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px;
    color: var(--cyan) !important;
    letter-spacing: 1.5px;
    opacity: 0.8;
    margin-top: 3px;
}

/* ─── GLASS CARD (sidebar panels) ─── */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    padding: 18px 16px;
    margin: 10px 0;
    position: relative;
    overflow: hidden;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    animation: cardEntrance 0.5s ease both;
}
.glass-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--cyan), var(--blue), var(--pink));
    opacity: 0.6;
}
.glass-card:hover {
    transform: translateY(-3px);
    border-color: rgba(12,242,179,0.4);
    box-shadow: 0 12px 40px rgba(12,242,179,0.15);
}
@keyframes cardEntrance {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ─── SECTION LABELS ─── */
.section-label {
    font-family: 'Orbitron', monospace;
    font-size: 10px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--cyan) !important;
    margin-bottom: 10px;
    opacity: 0.9;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--cyan-dim), transparent);
}

/* ─── SUMMARY BOX ─── */
.summary-box {
    background: linear-gradient(135deg, rgba(255,63,164,0.06), rgba(12,242,179,0.06));
    border: 1px solid rgba(255,63,164,0.25);
    border-radius: var(--radius);
    padding: 16px;
    color: var(--white-soft) !important;
    font-family: 'Exo 2', sans-serif;
    font-size: 13.5px;
    line-height: 1.7;
    position: relative;
    animation: summaryPulse 5s ease-in-out infinite;
}
.summary-box::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0; width: 3px;
    background: linear-gradient(180deg, var(--pink), var(--cyan));
    border-radius: 4px 0 0 4px;
}
@keyframes summaryPulse {
    0%,100% { box-shadow: 0 0 10px rgba(255,63,164,0.08); }
    50%      { box-shadow: 0 0 22px rgba(255,63,164,0.18); }
}

/* ─── STATUS DOTS ─── */
.status-panel {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    padding: 14px 16px;
    margin-top: 10px;
    animation: cardEntrance 0.6s 0.2s ease both;
}
.status-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 6px 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    color: var(--white-soft) !important;
}
.dot-online  { width: 7px; height: 7px; border-radius: 50%; background: var(--cyan);  animation: dotBlink 2s infinite; flex-shrink: 0; }
.dot-active  { width: 7px; height: 7px; border-radius: 50%; background: var(--blue);  animation: dotBlink 2s 0.5s infinite; flex-shrink: 0; }
.dot-stream  { width: 7px; height: 7px; border-radius: 50%; background: var(--gold);  animation: dotBlink 2s 1s infinite; flex-shrink: 0; }
@keyframes dotBlink {
    0%,100% { opacity: 1; box-shadow: 0 0 6px currentColor; }
    50%      { opacity: 0.4; box-shadow: none; }
}

/* ─── MAIN HEADING ─── */
.main-heading {
    font-family: 'Orbitron', monospace;
    font-size: clamp(18px, 3vw, 28px);
    font-weight: 900;
    text-align: center;
    letter-spacing: 3px;
    background: linear-gradient(120deg, var(--cyan), var(--blue), var(--pink));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    padding: 20px 24px;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    background-color: var(--bg-card);
    position: relative;
    overflow: hidden;
    animation: headingEntrance 0.6s ease both;
}
.main-heading::before {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--cyan), var(--blue), var(--pink));
}
.main-heading::after {
    content: '';
    position: absolute;
    top: 0; left: -60%;
    width: 40%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.04), transparent);
    animation: headingShine 4s ease-in-out infinite;
}
@keyframes headingEntrance {
    from { opacity: 0; transform: translateY(-16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes headingShine {
    0%   { left: -60%; }
    100% { left: 160%; }
}

.sub-heading {
    text-align: center;
    color: var(--grey) !important;
    font-family: 'Share Tech Mono', monospace;
    font-size: 11.5px;
    letter-spacing: 2px;
    margin-top: 8px;
    animation: headingEntrance 0.6s 0.15s ease both;
}
.sub-heading span { color: var(--cyan) !important; }

/* ─── IDLE STATE ─── */
.idle-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    padding: 52px 32px;
    text-align: center;
    margin: 40px auto;
    max-width: 520px;
    position: relative;
    overflow: hidden;
    animation: cardEntrance 0.5s ease both;
}
.idle-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--cyan), var(--blue), var(--pink));
}
.idle-icon {
    font-size: 54px;
    display: block;
    margin-bottom: 18px;
    animation: idleFloat 3s ease-in-out infinite;
    filter: drop-shadow(0 0 14px var(--cyan));
}
@keyframes idleFloat {
    0%,100% { transform: translateY(0); }
    50%      { transform: translateY(-8px); }
}
.idle-title {
    font-family: 'Orbitron', monospace;
    font-size: 16px;
    font-weight: 700;
    color: var(--cyan) !important;
    letter-spacing: 2px;
    margin-bottom: 10px;
}
.idle-sub {
    color: var(--grey) !important;
    font-size: 13px;
    line-height: 1.7;
    font-family: 'Exo 2', sans-serif;
}

/* ─── CHAT MESSAGES ─── */
.stChatMessage { background: transparent !important; }

/* User message */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    animation: slideRight 0.35s cubic-bezier(0.22, 1, 0.36, 1) both;
}
@keyframes slideRight {
    from { opacity: 0; transform: translateX(30px); }
    to   { opacity: 1; transform: translateX(0); }
}

.user-message {
    background: linear-gradient(135deg, rgba(12,242,179,0.1), rgba(75,217,255,0.07));
    border: 1px solid rgba(12,242,179,0.25);
    border-left: 3px solid var(--cyan);
    border-radius: 0 var(--radius) var(--radius) var(--radius);
    padding: 14px 18px;
    color: var(--white) !important;
    font-family: 'Exo 2', sans-serif;
    font-size: 14.5px;
    line-height: 1.65;
    position: relative;
    transition: border-color 0.25s;
}
.user-message:hover { border-color: rgba(12,242,179,0.5); }

/* Assistant message */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    animation: slideLeft 0.35s cubic-bezier(0.22, 1, 0.36, 1) both;
}
@keyframes slideLeft {
    from { opacity: 0; transform: translateX(-30px); }
    to   { opacity: 1; transform: translateX(0); }
}

.assistant-message {
    background: linear-gradient(135deg, rgba(255,63,164,0.08), rgba(12,242,179,0.06));
    border: 1px solid rgba(255,63,164,0.2);
    border-right: 3px solid var(--pink);
    border-radius: var(--radius) 0 var(--radius) var(--radius);
    padding: 14px 18px;
    color: var(--white) !important;
    font-family: 'Exo 2', sans-serif;
    font-size: 14.5px;
    line-height: 1.75;
    position: relative;
    transition: border-color 0.25s;
}
.assistant-message:hover { border-color: rgba(255,63,164,0.45); }
.assistant-message strong { color: var(--cyan) !important; font-weight: 700; }
.assistant-message em { color: var(--blue) !important; }
.assistant-message code {
    background: rgba(12,242,179,0.1);
    color: var(--cyan) !important;
    padding: 2px 7px;
    border-radius: 4px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 13px;
}

/* Streaming cursor */
.streaming-cursor {
    display: inline-block;
    width: 2px; height: 17px;
    background: var(--cyan);
    margin-left: 4px;
    vertical-align: middle;
    animation: cursorBlink 0.8s steps(1) infinite;
    box-shadow: 0 0 6px var(--cyan);
}
@keyframes cursorBlink {
    0%,50% { opacity: 1; }
    51%,100% { opacity: 0; }
}

/* ─── CHAT INPUT ─── */
[data-testid="stChatInput"] > div {
    background: var(--bg-card) !important;
    border: 1px solid rgba(75,217,255,0.3) !important;
    border-radius: var(--radius) !important;
    transition: border-color 0.25s, box-shadow 0.25s;
    animation: inputPulse 4s ease-in-out infinite;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 20px rgba(12,242,179,0.2) !important;
    animation: none;
}
@keyframes inputPulse {
    0%,100% { border-color: rgba(75,217,255,0.3) !important; }
    50%      { border-color: rgba(12,242,179,0.45) !important; }
}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] input {
    color: var(--white) !important;
    font-family: 'Exo 2', sans-serif !important;
    font-size: 14px !important;
    background: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--grey) !important;
}

/* ─── BUTTONS ─── */
.stButton > button {
    background: linear-gradient(135deg, rgba(12,242,179,0.12), rgba(75,217,255,0.08)) !important;
    border: 1px solid rgba(12,242,179,0.4) !important;
    color: var(--cyan) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 10.5px !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    border-radius: var(--radius-sm) !important;
    padding: 10px 18px !important;
    transition: all 0.25s ease !important;
    position: relative;
    overflow: hidden;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(12,242,179,0.22), rgba(75,217,255,0.15)) !important;
    border-color: var(--cyan) !important;
    box-shadow: 0 0 20px rgba(12,242,179,0.3) !important;
    transform: translateY(-2px) !important;
    color: #fff !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, rgba(255,63,164,0.12), rgba(244,196,48,0.08)) !important;
    border: 1px solid rgba(255,63,164,0.4) !important;
    color: var(--pink) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 10px !important;
    letter-spacing: 1px !important;
    border-radius: var(--radius-sm) !important;
    transition: all 0.25s ease !important;
}
.stDownloadButton > button:hover {
    box-shadow: 0 0 20px rgba(255,63,164,0.3) !important;
    transform: translateY(-2px) !important;
}

/* ─── FILE UPLOADER ─── */
[data-testid="stFileUploader"] {
    background: rgba(12,242,179,0.03) !important;
    border: 1px dashed rgba(12,242,179,0.3) !important;
    border-radius: var(--radius) !important;
    padding: 8px !important;
    transition: border-color 0.25s, background 0.25s;
}
[data-testid="stFileUploader"]:hover {
    background: rgba(12,242,179,0.06) !important;
    border-color: rgba(12,242,179,0.55) !important;
}
[data-testid="stFileUploader"] * { color: var(--white-soft) !important; }

/* ─── EXPANDER (source docs) ─── */
.streamlit-expanderHeader {
    background: rgba(12,242,179,0.06) !important;
    border: 1px solid rgba(12,242,179,0.2) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--cyan) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 12px !important;
    transition: all 0.2s;
}
.streamlit-expanderHeader:hover {
    background: rgba(12,242,179,0.12) !important;
    border-color: rgba(12,242,179,0.4) !important;
}
.streamlit-expanderContent {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius-sm) var(--radius-sm) !important;
    padding: 10px !important;
}

/* ─── DIVIDER ─── */
hr { border-color: var(--border-subtle) !important; margin: 14px 0 !important; }

/* ─── SPINNER ─── */
.stSpinner > div { border-top-color: var(--cyan) !important; }

/* ─── ALERTS ─── */
.stAlert {
    background: rgba(255,63,164,0.08) !important;
    border: 1px solid rgba(255,63,164,0.3) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--white-soft) !important;
}
.stSuccess {
    background: rgba(12,242,179,0.08) !important;
    border: 1px solid rgba(12,242,179,0.35) !important;
    color: var(--white-soft) !important;
}

/* ─── CAPTION ─── */
.stCaptionContainer, .stCaption, caption, small {
    color: var(--grey) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 11px !important;
}

/* ─── MAIN AREA PADDING ─── */
.block-container { padding-top: 24px !important; }

/* ─── PROGRESS BAR ─── */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--cyan), var(--blue), var(--pink)) !important;
}

/* ─── METRIC BOXES (stat pills) ─── */
.stat-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 100px;
    padding: 5px 14px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    color: var(--white-soft) !important;
    transition: all 0.25s;
    animation: pillEntrance 0.4s ease both;
}
.stat-pill:hover {
    border-color: rgba(12,242,179,0.4);
    box-shadow: 0 0 12px rgba(12,242,179,0.15);
}
@keyframes pillEntrance {
    from { opacity: 0; transform: scale(0.85); }
    to   { opacity: 1; transform: scale(1); }
}

/* Make main content text visible */
.stMarkdown p, .stMarkdown li, .stMarkdown span,
.element-container p, .element-container span {
    color: var(--white-soft) !important;
}
</style>

<script>
// Auto-scroll on new messages
const autoScroll = () => {
    const main = document.querySelector('.main .block-container');
    if (main) main.scrollTo({ top: main.scrollHeight, behavior: 'smooth' });
};
const obs = new MutationObserver(autoScroll);
const target = document.querySelector('.main');
if (target) obs.observe(target, { childList: true, subtree: true });
</script>
""", unsafe_allow_html=True)

# ==================== LOGO (MAIN AREA) ====================
st.markdown("""
<div class="ss-logo-wrap">
    <div class="ss-logo-icon">⚡</div>
    <div>
        <div class="ss-logo-text">S.S_AI</div>
        <div class="ss-logo-tag">NEURAL PDF ASSISTANT &nbsp;·&nbsp; v2.0</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==================== BACKEND ====================
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

if "messages"          not in st.session_state: st.session_state.messages = []
if "chain"             not in st.session_state: st.session_state.chain = None
if "summary"           not in st.session_state: st.session_state.summary = None
if "memory"            not in st.session_state: st.session_state.memory = None
if "last_request_time" not in st.session_state: st.session_state.last_request_time = 0
if "pdf_count"         not in st.session_state: st.session_state.pdf_count = 0

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

@retry_on_rate_limit
def invoke_with_retry(llm, prompt):
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

    time.sleep(1)
    try:
        response = invoke_with_retry(llm, prompt)
        return response.content
    except Exception:
        st.error("⚠️ Rate limit reached. Please wait a moment and try again.")
        return "Summary temporarily unavailable. Please try again in a few seconds."

# ==================== MAIN HEADING ====================
st.markdown('<div class="main-heading">📚 &nbsp; NEURAL PDF INTERFACE</div>', unsafe_allow_html=True)
st.markdown('<p class="sub-heading">⚡ POWERED BY <span>GROQ LLaMA 3.3 70B</span> &nbsp;|&nbsp; ULTRA-FAST INFERENCE &nbsp;|&nbsp; BILINGUAL SUPPORT</p>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:

    # Upload Card
    st.markdown("""
    <div class="glass-card">
        <div class="section-label">📄 &nbsp; UPLOAD INTERFACE</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Select PDF Files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("⚡  INITIATE NEURAL PROCESSING", use_container_width=True):
        if not uploaded_files:
            st.warning("⚠️ No PDF detected — please upload files first.")
        else:
            with st.spinner("🔮 Building vector index…"):
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
                st.session_state.pdf_count = len(uploaded_files)

            with st.spinner("📊 Generating summary…"):
                st.session_state.summary = generate_summary(vector_store)

            st.success(f"✅ {len(uploaded_files)} PDF(s) integrated successfully!")

    # Summary
    if st.session_state.summary:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="section-label" style="color: #FF3FA4 !important;">
            📋 &nbsp; NEURAL SUMMARY
            <span style="flex:1; height:1px; background: linear-gradient(90deg, rgba(255,63,164,0.3), transparent); display:inline-block;"></span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f'<div class="summary-box">{st.session_state.summary}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)

    # Export
    if st.session_state.messages:
        st.markdown('<div class="section-label">💾 &nbsp; EXPORT</div>', unsafe_allow_html=True)
        if st.button("📁  EXPORT CHAT LOG", use_container_width=True):
            export_data = [{"role": m["role"], "message": m["content"]} for m in st.session_state.messages]
            filename = f"neural_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
            st.download_button(
                label="⬇️  DOWNLOAD JSON",
                data=json.dumps(export_data, ensure_ascii=False, indent=2),
                file_name=filename,
                mime="application/json",
                use_container_width=True,
            )
        st.markdown("<br>", unsafe_allow_html=True)

    # Status panel
    st.markdown("""
    <div class="status-panel">
        <div class="section-label">🖥️ &nbsp; SYSTEM STATUS</div>
        <div class="status-row"><span class="dot-online"></span> AI CORE: ONLINE</div>
        <div class="status-row"><span class="dot-active"></span> MODEL: LLaMA 3.3 70B</div>
        <div class="status-row"><span class="dot-stream"></span> STREAMING: ACTIVE</div>
    </div>
    """, unsafe_allow_html=True)

    # Stats pills if PDF loaded
    if st.session_state.pdf_count:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:flex; gap:8px; flex-wrap:wrap;">
            <span class="stat-pill">📄 {st.session_state.pdf_count} PDF(s)</span>
            <span class="stat-pill">💬 {len(st.session_state.messages)//2} turns</span>
        </div>
        """, unsafe_allow_html=True)

# ==================== CHAT AREA ====================
if not st.session_state.chain:
    st.markdown("""
    <div class="idle-card">
        <span class="idle-icon">🧠</span>
        <div class="idle-title">NEURAL INTERFACE READY</div>
        <div class="idle-sub">
            Upload your PDF documents in the sidebar<br>
            and click <strong style="color:#0CF2B3;">Initiate Neural Processing</strong><br>
            to begin intelligent Q&A.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Display history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f'<div class="user-message">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        with st.chat_message("assistant"):
            st.markdown(f'<div class="assistant-message">{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander("📖  SOURCE DOCUMENTS"):
                    for src in msg["sources"]:
                        st.caption(f"📄 {src}")

# Chat input
if user_input := st.chat_input("⚡ Enter your query… (Urdu | English)"):
    with st.chat_message("user"):
        st.markdown(f'<div class="user-message">{user_input}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        current_time = time.time()
        if current_time - st.session_state.last_request_time < 2:
            st.warning("⚠️ Please wait a moment between messages (rate limit protection)")
            time.sleep(2)

        with st.spinner("🧠 Processing…"):
            try:
                result = st.session_state.chain.invoke({"question": user_input})
                answer = result["answer"]
                st.session_state.last_request_time = time.time()

                placeholder = st.empty()
                full_response = ""
                for word in answer.split():
                    full_response += word + " "
                    placeholder.markdown(
                        f'<div class="assistant-message">{full_response}<span class="streaming-cursor"></span></div>',
                        unsafe_allow_html=True
                    )
                    time.sleep(0.015)
                placeholder.markdown(f'<div class="assistant-message">{full_response}</div>', unsafe_allow_html=True)

                sources, seen = [], set()
                for doc in result["source_documents"]:
                    src = f"{doc.metadata.get('file_name', '?')}  ·  PAGE {doc.metadata.get('page', 0) + 1}"
                    if src not in seen:
                        sources.append(src)
                        seen.add(src)

                if sources:
                    with st.expander("📖  SOURCE DOCUMENTS"):
                        for src in sources:
                            st.caption(f"📄 {src}")

            except Exception:
                answer = "⚠️ Too many requests. Please wait a moment and try again."
                st.markdown(f'<div class="assistant-message">{answer}</div>', unsafe_allow_html=True)
                sources = []

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })

    st.rerun()
