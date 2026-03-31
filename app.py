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

# --- Page config ---

st.set_page_config(page_title="PDF Talker 2.0", page_icon="📚", layout="wide")
st.markdown("""

<style>
/* Minimal Elegant UI */
body {
    background-color: #f4f4f9;
    color: #222222;
    font-family: 'Segoe UI', sans-serif;
}
.stApp {
    background-color: #f4f4f9;
}
.stButton>button {
    background-color: #4BD9FF;
    color: white;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
}
.stButton>button:hover {
    background-color: #33c3ff;
}
.stFileUploader>div {
    border: 2px dashed #cccccc;
    border-radius: 8px;
    padding: 10px;
    background-color: #ffffff;
}
h1, h2, h3 {
    background-color: #e0f7ff;
    padding: 6px 12px;
    border-radius: 6px;
}
.stDivider {
    border-top: 1px solid #cccccc;
}
.stChatMessage {
    background-color: #ffffff;
    padding: 8px 12px;
    border-radius: 8px;
    margin-bottom: 4px;
}
</style>

""", unsafe_allow_html=True)

st.title("📚 PDF Talker 2.0")
st.caption("Powered by Groq LLaMA 3.3 70B · Professional Urdu/English Support")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# --- Session State ---

if "messages" not in st.session_state:
st.session_state.messages = []
if "chain" not in st.session_state:
st.session_state.chain = None
if "summary" not in st.session_state:
st.session_state.summary = None
if "memory" not in st.session_state:
st.session_state.memory = None

# --- Embeddings ---

@st.cache_resource
def get_embeddings():
return HuggingFaceEmbeddings(
model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
model_kwargs={"device": "cpu"},
encode_kwargs={"normalize_embeddings": True},
)

# --- Build Chain ---

def build_chain(vector_store):
SYSTEM_PROMPT = """You are a professional PDF assistant with expert-level understanding.

Rules:

1. Answer from PDF only.
2. Maintain language consistency.
3. Cite page numbers.
   """

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

   retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k":8,"fetch_k":25,"lambda_mult":0.5})

   return ConversationalRetrievalChain.from_llm(
   llm=llm,
   retriever=retriever,
   memory=memory,
   combine_docs_chain_kwargs={"prompt": qa_prompt},
   return_source_documents=True,
   verbose=False,
   )

# --- Summary Generation ---

def generate_summary(vector_store):
docs = vector_store.similarity_search("main topics key points overview summary", k=5)
context = "\n\n".join([d.page_content for d in docs])
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, api_key=GROQ_API_KEY)
prompt = f"""Generate a 4-5 line summary in the same language as PDF.
Content:\n{context}"""
response = llm.invoke(prompt)
return response.content

# --- Sidebar ---

with st.sidebar:
st.header("📄 PDF Upload")
uploaded_files = st.file_uploader("Select PDFs", type=["pdf"], accept_multiple_files=True)

```
if st.button("⚡ Process PDFs", use_container_width=True):
    if not uploaded_files:
        st.warning("Please select PDFs!")
    else:
        with st.spinner("Processing PDFs..."):
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

            splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
            chunks = splitter.split_documents(all_pages)

            vector_store = FAISS.from_documents(documents=chunks, embedding=get_embeddings())
            st.session_state.chain = build_chain(vector_store)
            st.session_state.messages = []

        with st.spinner("Generating summary..."):
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
        st.download_button(label="⬇️ Download JSON", data=json.dumps(export_data, ensure_ascii=False, indent=2), file_name=filename, mime="application/json", use_container_width=True)
```

# --- Chat Area ---

if not st.session_state.chain:
st.info("👈 Upload PDFs from sidebar and click 'Process PDFs'")
st.stop()

for msg in st.session_state.messages:
with st.chat_message(msg["role"]):
st.markdown(msg["content"])
if msg.get("sources"):
with st.expander("📖 Sources"):
for src in msg["sources"]:
st.caption(f"📄 {src}")

if user_input := st.chat_input("Ask your question... (Urdu or English)"):
with st.chat_message("user"):
st.markdown(user_input)
st.session_state.messages.append({"role": "user", "content": user_input})

```
with st.chat_message("assistant"):
    with st.spinner("Generating response..."):
        result = st.session_state.chain.invoke({"question": user_input})
        answer = result["answer"]

        # Stream response
        response_placeholder = st.empty()
        full_response = ""
        for word in answer.split():
            full_response += word + " "
            response_placeholder.markdown(full_response + "▌")
            time.sleep(0.015)
        response_placeholder.markdown(full_response)

        # Sources
        sources = []
        seen = set()
        for doc in result["source_documents"]:
            src = f"{doc.metadata.get('file_name', '?')} · Page {doc.metadata.get('page', 0) + 1}"
            if src not in seen:
                sources.append(src)
                seen.add(src)

        if sources:
            with st.expander("📖 Source Documents"):
                for src in sources:
                    st.caption(f"📄 {src}")

st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
st.rerun()
```
