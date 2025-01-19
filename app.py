from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from sqlalchemy import create_engine

from plot import plot_csv_data, plot_dataframe_data
from tools import execute_query_tool
from audio import audiototextOpenAI

import os
os.makedirs('./data', exist_ok=True)


CSV_OUTPUT_DIR = "./output"
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
CSV_FILE_PATH = os.path.join(CSV_OUTPUT_DIR, 'csv_results.csv')



load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
## Langmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
# os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")
# os.environ['MISTRAL_API_KEY'] = os.getenv("MISTRAL_API_KEY")


GPT4 = ChatOpenAI(model_name="gpt-4", temperature=0)


URI = "./milvusdb.db"
from langchain_milvus import Milvus
# vector_store = Milvus(
#     embedding_function=st.session_state.embeddings=OpenAIEmbeddings(),
#     connection_args={"uri": URI},
# )

st.session_state['vectorDB'] = False

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=OpenAIEmbeddings()
        st.session_state.loader=DirectoryLoader("./data",   glob="**/*.txt", loader_cls=TextLoader, use_multithreading=True ) ## Data Ingestion
        st.session_state.docs=st.session_state.loader.load() ## Documents Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors=Milvus.from_documents(st.session_state.final_documents,
                                                       st.session_state.embeddings,
                                                       collection_name="Olist",
                                                       connection_args={"uri": URI},
                                                        drop_old=True)
        # return st.session_state.vectors


import pandas as pd

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)


# os.environ['POSTGRES_DB_URL'] = os.getenv("POSTGRES_DB_URL")
db_url = os.getenv("POSTGRES_DB_URL")


with st.sidebar:
    st.write("Llama3.3")    
    if st.button("Connect to DB"):
        engine = create_engine(db_url)
        try:
            with engine.connect() as connection:
                st.success("Connection successful")
        except Exception as e:  
            st.write(f"Failed to connect to the database: {e}")    

    st.session_state["upload_status"] = False
    uploaded_files = st.sidebar.file_uploader("Upload Files to Vectorize", accept_multiple_files=True)
    if uploaded_files:
        # for file in uploaded_files:
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            file_path = os.path.join('./data', uploaded_file.name)

            with open(file_path, 'wb') as file:
                file.write(bytes_data)
        
            st.session_state["upload_status"] = True


    if st.session_state["upload_status"]:
        if st.button("Load Documents"):
            vector_embedding()
            st.write("Documents Loaded Successfully")
            st.session_state['vectorDB'] = True

llm = ChatNVIDIA(
  model="meta/llama-3.3-70b-instruct",
  api_key="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", 
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
)



st.title("InsightIQ")
st.caption("🚀 Unleash Private AI: LLMs Running Securely on Your Device - Powered by MistralAI and NVIDIA NIM")

input_prompt = None
if st.session_state['vectorDB']:
    audio_value = st.audio_input("Record a voice message", label_visibility="hidden")
    audiototextOpenAI(audio_value)
    # input_prompt=st.text_input("Enter Your Question From Documents", key="userinput")
    input_prompt = audiototextOpenAI(audio_value)
    st.write(input_prompt.text)

import re
import pandas as pd
db_url = 'postgresql://postgres:admin@localhost:5432/olist'
engine = create_engine(db_url)

st.session_state["sql_query_match"] = False
st.session_state["insights_gen"] = False

if input_prompt  and st.session_state.vectors is not None:
    sql_query_prompt = f"""
    Based on the provided schema and table details, translate the following natural language query into an SQL query. Ensure the SQL query is as accurate as possible to answer the user’s request and provide an explanation of the query's significance in general. The SQL query should be enclosed within triple backticks and the `sql` keyword. If you cannot generate an SQL query, respond with 'No SQL query generated'.
    Natural Language Query: {input_prompt}
    """
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    response=retrieval_chain.invoke({'input':sql_query_prompt})

        # Write input_prompt and response['answer'] to queryresults.txt
    with open("queryresults.txt", "a") as file:
        file.write(f"Query: {input_prompt}\n")
        file.write(f"Answer: {response['answer']}\n")
        file.write("\n" + "-"*50 + "\n\n")
    
    st.write(response['answer'])
    # print(response['answer'])

    # /query_match = re.search(r"```sql(.*?)```", response["answer"], re.DOTALL)
    query_match = re.search(r"```sql(.*?)```|SELECT\s+.*?;", response["answer"], re.DOTALL | re.IGNORECASE)

    if query_match:
        sql_query = query_match.group(1).strip()
        # print("Extracted SQL Query:")
        # print(sql_query)

        df = execute_query_tool(sql_query)
        data_str = df.to_string(index=False)

        analysis_prompt = f"Analyze the following data: as you were the Head of Sales and Marketing, the following strategic actions would be based on the results to drive business growth and maximize returns:\n{data_str}"

        insightresult = ""
        for chunk in llm.stream([{"role": "user", "content": analysis_prompt}]):
            insightresult += chunk.content
        
        # print(insightresult)
        st.write(insightresult)
        st.session_state["sql_query_match"] = True

        st.dataframe(df)
    else:
        print("No SQL query found in the response.")

