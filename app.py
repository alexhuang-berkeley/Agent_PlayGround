import os
import io
import pandas as pd
import streamlit as st
import sqlalchemy
import tempfile

try:
    from langchain.llms.bedrock import Bedrock
    from langchain.agents import initialize_agent, AgentType, Tool
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import SQLDatabaseChain
    from langchain.sql_database import SQLDatabase
    from langchain.prompts import PromptTemplate
except Exception:
    Bedrock = None

# Optional dependencies
try:
    import docx2txt
except ImportError:
    docx2txt = None

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

# Setup page
st.set_page_config(page_title="Single-Agent LLM Playground", layout="wide")

st.title("Single-Agent LLM Playground")

# Sidebar for uploads and prompt editing
with st.sidebar:
    st.header("Uploads")
    uploaded_docs = st.file_uploader(
        "Upload documents", type=["pdf", "txt", "docx"], accept_multiple_files=True
    )
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

    def doc_tool(_: str) -> str:
        if uploaded_docs:
            texts = [load_document(f) for f in uploaded_docs]
            return "\n".join(texts)
        return ""

    tools.append(
        Tool(
            name="document_context",
            func=doc_tool,
            description="Return the text of uploaded documents",
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

user_msg = st.chat_input("Ask something")
if user_msg:
    if agent:
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
if st.checkbox("Show raw history"):
    st.json(st.session_state.history)
if steps and st.checkbox("Show intermediate steps"):
    st.json(steps)
