import os
import io
import pandas as pd
import streamlit as st
import sqlalchemy
import tempfile

# Import LangChain components used to build the vector database and agent
from langchain.docstore.document import Document  # container for text passages
from langchain.text_splitter import RecursiveCharacterTextSplitter  # split documents into chunks
from langchain.embeddings.openai import OpenAIEmbeddings  # create embeddings for text
from langchain.vectorstores import FAISS  # in-memory vector store for similarity search

try:
    from langchain.llms.bedrock import Bedrock
    from langchain.agents import initialize_agent, AgentType, Tool
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import SQLDatabaseChain
    from langchain.sql_database import SQLDatabase
    from langchain.prompts import PromptTemplate
except Exception:
    # LangChain extras may not be available in minimal environments
    Bedrock = None

# Optional dependencies for document parsing
try:
    import docx2txt
except ImportError:
    docx2txt = None

try:
    from PyPDF2 import PdfReader  # used for extracting text from PDFs
except ImportError:
    PdfReader = None  # PDF support is optional

# Setup page
st.set_page_config(page_title="Single-Agent LLM Playground", layout="wide")

st.title("Single-Agent LLM Playground")

# Sidebar for uploads and prompt editing
with st.sidebar:
    st.header("Uploads")
    # Allow multiple document uploads. When the set of files changes we rebuild
    # the vector index so queries see the latest content.
    uploaded_docs = st.file_uploader(
        "Upload documents", type=["pdf", "txt", "docx"], accept_multiple_files=True
    )
    if uploaded_docs:
        doc_names = [f.name for f in uploaded_docs]
        if (
            "doc_names" not in st.session_state
            or st.session_state.doc_names != doc_names
        ):
            # Read each file and pull out its text
            texts = [load_document(f) for f in uploaded_docs]
            # Wrap raw strings in LangChain Document objects
            docs = [Document(page_content=t) for t in texts]
            # Break large documents into smaller chunks for embedding
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = splitter.split_documents(docs)
            # Create embeddings and store them in a FAISS index
            embeddings = OpenAIEmbeddings()
            st.session_state.doc_index = FAISS.from_documents(splits, embeddings)
            st.session_state.doc_names = doc_names
    # Single CSV upload is loaded into an in-memory SQLite database
    uploaded_csv = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
    if uploaded_csv:
        if "csv_name" not in st.session_state or st.session_state.csv_name != uploaded_csv.name:
            df_tmp = pd.read_csv(uploaded_csv)
            st.session_state.csv_df = df_tmp
            st.session_state.csv_name = uploaded_csv.name
            if Bedrock and SQLDatabase:
                engine = sqlalchemy.create_engine("sqlite:///:memory:")
                df_tmp.to_sql("data", engine, index=False, if_exists="replace")
                st.session_state.sql_engine = engine
                st.session_state.sql_db = SQLDatabase(engine)

    st.header("Prompts")
    system_prompt = st.text_area(
        "System Prompt", value="You are a helpful assistant.", key="system_prompt"
    )
    schema_prompt = st.text_area(
        "Schema Prompt", value="", key="schema_prompt"
    )
    st.divider()
    model = st.selectbox(
        "Model",
        [
            "anthropic.claude-instant-v1",
            "anthropic.claude-v2",
        ],
        key="model_select",
    )
    memory_toggle = st.checkbox("With Memory", value=True, key="memory_toggle")
    if st.button("Reset Chat"):
        st.session_state.history = []

# Helper functions

def load_document(file):
    if file.type == "application/pdf" and PdfReader:
        reader = PdfReader(file)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text
    if file.type == "text/plain":
        return file.getvalue().decode("utf-8")
    if file.type in [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ] and docx2txt:
        return docx2txt.process(file)
    return ""


def create_langchain_agent(llm, memory, system_prompt: str, schema_prompt: str):
    """Create a LangChain agent with tools for SQL and document context."""
    tools = []

    if Bedrock and SQLDatabase and "sql_db" in st.session_state:
        # Build a tool that lets the agent run SQL against the uploaded CSV
        db = st.session_state.sql_db

        def sql_tool(query: str) -> str:
            sql_prompt = PromptTemplate(
                input_variables=["input", "table_info", "dialect"],
                template=f"{schema_prompt}\n{{input}}",
            )
            chain = SQLDatabaseChain.from_llm(
                llm, db, verbose=False, prompt=sql_prompt
            )
            try:
                return chain.run(query)
            except Exception as e:
                return f"SQL error: {e}"

        tools.append(
            Tool(
                name="query_sql",
                func=sql_tool,
                description="Execute arbitrary SQL on the uploaded CSV",
            )
        )

    def doc_tool(query: str) -> str:
        """Search the FAISS index for passages relevant to the query."""
        if "doc_index" in st.session_state:
            docs = st.session_state.doc_index.similarity_search(query, k=4)
            # Concatenate the retrieved text chunks for the agent
            return "\n".join(d.page_content for d in docs)
        return ""

    tools.append(
        Tool(
            name="document_context",
            func=doc_tool,
            description="Search uploaded documents. Input is your query",
        )
    )

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs={"prefix": system_prompt},
        verbose=False,
        memory=memory,
        handle_parsing_errors=True,
    )
    return agent



# Chat area
st.header("Chat")

if "history" not in st.session_state:
    st.session_state.history = []

for h in st.session_state.history:
    with st.chat_message(h["role"]):
        st.write(h["content"])

llm = None
steps = []
if Bedrock:
    import boto3
    region = os.environ.get("AWS_REGION", "us-east-1")
    client = boto3.client("bedrock-runtime", region_name=region)
    llm = Bedrock(client=client, model_id=model)

memory = ConversationBufferMemory(memory_key="chat_history") if memory_toggle else None
agent = create_langchain_agent(llm, memory, system_prompt, schema_prompt) if llm else None

# Main chat input box. The agent will see this message and reply.
user_msg = st.chat_input("Ask something")
if user_msg:
    if agent:
        # Run the query through the agent and collect intermediate reasoning steps
        result = agent.invoke({"input": user_msg})
        answer = result.get("output", "")
        steps = result.get("intermediate_steps", [])
    else:
        answer = "LangChain not available."
    st.session_state.history.append({"role": "user", "content": user_msg})
    st.session_state.history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)

st.header("Chain of Thought")
# Optional debug panels to inspect how the agent arrived at its answer
if st.checkbox("Show raw history"):
    st.json(st.session_state.history)
if steps and st.checkbox("Show intermediate steps"):
    st.json(steps)
