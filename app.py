import os, re
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Zyro HR Help Desk", page_icon="💼")

load_dotenv()
load_dotenv(Path(__file__).resolve().parent / ".venv" / ".env")
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY is missing. Add it to .env, .venv/.env, or the app environment.")
    st.stop()

if "LANGSMITH_API_KEY" in st.secrets:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
    os.environ["LANGSMITH_PROJECT"] = "zyro-hr-helpdesk"

K, FETCH_K, LAMBDA = 4, 20, 0.6
LOW, HIGH = 0.40, 0.60            # use the same values you tuned in the notebook
LLM_MODEL = "llama-3.3-70b-versatile"
REFUSAL = "I can only help with Zyro Dynamics HR policy questions. Please ask me about leave, WFH, travel, compensation, conduct and similar topics."


def load_vector_store(embeddings):
    project_dir = Path(__file__).resolve().parent
    index_dir = project_dir / "faiss_index"
    index_file = index_dir / "index.faiss"
    metadata_file = index_dir / "index.pkl"
    if index_file.is_file() and metadata_file.is_file():
        return FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)

    corpus_archive = project_dir / "gen-ai-project-intelligent-rag(1).zip"
    if not corpus_archive.is_file():
        st.error(f"FAISS index is missing and the PDF corpus archive was not found: {corpus_archive}")
        st.stop()

    documents = []
    with ZipFile(corpus_archive) as archive:
        for member in archive.infolist():
            if member.is_dir() or Path(member.filename).suffix.lower() != ".pdf":
                continue
            policy = re.sub(r"^\d+_", "", Path(member.filename).stem).replace("_", " ")
            pdf = PdfReader(BytesIO(archive.read(member)))
            for page_no, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={"policy": policy, "page_no": page_no},
                    ))

    if not documents:
        st.error(f"No readable PDF pages were found in {corpus_archive}.")
        st.stop()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    ).split_documents(documents)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(index_dir))
    return vector_store


@st.cache_resource(show_spinner="Loading HR knowledge base...")
def load_all():
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5",
                                encode_kwargs={"normalize_embeddings": True})
    vs = load_vector_store(emb)
    chunks = list(vs.docstore._dict.values())
    tok = lambda t: re.findall(r"[a-z0-9₹%]+", t.lower())
    bm25 = BM25Okapi([tok(c.page_content) for c in chunks])
    llm = ChatGroq(model=LLM_MODEL, temperature=0, api_key=groq_api_key)
    return vs, chunks, bm25, tok, llm


vs, chunks, bm25, tok, llm = load_all()
mmr = vs.as_retriever(search_type="mmr", search_kwargs={"k": K, "fetch_k": FETCH_K, "lambda_mult": LAMBDA})


def hybrid_retrieve(q, k=K):
    mmr_docs = mmr.invoke(q)
    sc = bm25.get_scores(tok(q))
    idx = sorted(range(len(sc)), key=lambda i: sc[i], reverse=True)[:8]
    fused, by_key = {}, {}
    for ranked in (mmr_docs, [chunks[i] for i in idx]):
        for r, d in enumerate(ranked):
            key = (d.metadata["policy"], d.metadata["page_no"], d.page_content[:60])
            by_key[key] = d
            fused[key] = fused.get(key, 0) + 1 / (60 + r + 1)
    return [by_key[x] for x in sorted(fused, key=fused.get, reverse=True)[:k]]


def format_docs(docs):
    return "\n\n".join(f"[Source: {d.metadata['policy']}, page {d.metadata['page_no']}]\n{d.page_content}" for d in docs)


answer_prompt = ChatPromptTemplate.from_template(
"""You are the HR Help Desk assistant for Zyro Dynamics.
Answer the employee's question ONLY using the policy context below.
- Be clear and short. Use the exact numbers/rules from the context.
- After each fact, cite it like (Policy Name, page X).
- If the context does not contain the answer, reply exactly:
  "I don't have this in the HR policies. Please contact HR."
- Never use outside knowledge.

Context:
{context}

Question: {question}

Answer:""")

rag_chain = (
    RunnableParallel(docs=RunnableLambda(hybrid_retrieve), question=RunnablePassthrough())
    | RunnablePassthrough.assign(
        answer=RunnableLambda(lambda x: {"context": format_docs(x["docs"]), "question": x["question"]})
               | answer_prompt | llm | StrOutputParser())
)

scope_chain = ChatPromptTemplate.from_template(
"""Is this question about company HR policy (leave, work from home, travel and expenses, compensation,
code of conduct, benefits, holidays, onboarding, performance, etc.)?
Answer with exactly one word: YES or NO.

Question: {question}""") | llm | StrOutputParser()

rewrite_chain = ChatPromptTemplate.from_template(
"""Given the chat history and a follow-up question, rewrite the follow-up as a standalone question.
If it is already standalone, return it unchanged. Return only the question.

History:
{history}

Follow-up: {question}

Standalone question:""") | llm | StrOutputParser()


def best_similarity(q):
    d = vs.similarity_search_with_score(q, k=1)[0][1]
    return 1 - d ** 2 / 2


def is_in_scope(q):
    sim = best_similarity(q)
    if sim >= HIGH:
        return True, sim
    if sim < LOW:
        return False, sim
    return scope_chain.invoke({"question": q}).strip().upper().startswith("YES"), sim


def confidence_label(sim):
    return "High" if sim >= 0.70 else "Medium" if sim >= 0.55 else "Low"


def hr_email_draft(q):
    return (f"Subject: Policy clarification request\n\nHi HR Team,\n\n"
            f"I could not find a clear answer to the following in the HR policies:\n\n\"{q}\"\n\n"
            f"Could you please clarify?\n\nThank you.")


def ask(question, history):
    standalone = question
    if history:
        h = "\n".join(f"{r}: {t}" for r, t in history[-6:])
        standalone = rewrite_chain.invoke({"history": h, "question": question}).strip()
    ok, sim = is_in_scope(standalone)
    if not ok:
        return {"answer": REFUSAL, "sources": [], "conf": None, "email": None}
    out = rag_chain.invoke(standalone)
    unsure = "don't have this in the hr policies" in out["answer"].lower()
    conf = "Low" if unsure else confidence_label(sim)
    return {"answer": out["answer"], "sources": out["docs"], "conf": conf,
            "email": hr_email_draft(question) if conf == "Low" else None}


# ---------------- UI ----------------
st.title("💼 Zyro Dynamics HR Help Desk")
st.caption("Answers come only from the official HR policy PDFs, and every answer shows its source.")

if "msgs" not in st.session_state:
    st.session_state.msgs = []

for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("extra"):
            e = m["extra"]
            if e["conf"]:
                st.markdown(f"**Confidence:** {e['conf']}")
            for d in e["sources"]:
                with st.expander(f"📄 {d.metadata['policy']} (page {d.metadata['page_no']})"):
                    st.write(d.page_content)
            if e["email"]:
                st.info("Not sure about this one. You can send this to HR:")
                st.code(e["email"])

q = st.chat_input("Ask an HR policy question, e.g. How many casual leaves do I get?")
if q:
    history = [(("User" if m["role"] == "user" else "Assistant"), m["content"]) for m in st.session_state.msgs]
    st.session_state.msgs.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            r = ask(q, history)
        st.markdown(r["answer"])
        if r["conf"]:
            st.markdown(f"**Confidence:** {r['conf']}")
        for d in r["sources"]:
            with st.expander(f"📄 {d.metadata['policy']} (page {d.metadata['page_no']})"):
                st.write(d.page_content)
        if r["email"]:
            st.info("Not sure about this one. You can send this to HR:")
            st.code(r["email"])
    st.session_state.msgs.append({"role": "assistant", "content": r["answer"], "extra": r})