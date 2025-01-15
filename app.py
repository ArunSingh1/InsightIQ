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

from plot import plot_csv_data

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
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")
os.environ['MISTRAL_API_KEY'] = os.getenv("MISTRAL_API_KEY")
## Prompt Template

# prompt=ChatPromptTemplate.from_messages(
#     [
#         ("system","You are a helpful assistant. Please response to the user queries"),
#         ("user","Question:{question}")
#     ]
# )


GPT4 = ChatOpenAI(model_name="gpt-4", temperature=0)

Llama3_3 = ChatNVIDIA(
  model="meta/llama-3.3-70b-instruct",
  api_key="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", 
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
)

Mistral8b = ChatNVIDIA(
  model="mistralai/mistral-7b-instruct-v0.3",
  api_key="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", 
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
)

URI = "./milvusdb.db"
from langchain_milvus import Milvus
# vector_store = Milvus(
#     embedding_function=st.session_state.embeddings=OpenAIEmbeddings(),
#     connection_args={"uri": URI},
# )

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


def get_model(model_name):
    models = {
        "GPT4":GPT4,  # Replace with actual model initialization
        # "Llama3.3": Llama3_3 ,  # Replace with actual model initialization
        # "Mistral8b": Mistral8b  # Replace with actual model initialization
    }
    return models.get(model_name, None)


db_url = 'postgresql://postgres:admin@localhost:5432/olist'
engine = create_engine(db_url)
import psycopg2
import csv
def execute_query_tool(sql_query):
    try:
        # Connect to your postgres DB

        formatted_sql_query = f"""{sql_query}"""
        conn = psycopg2.connect(db_url)
        # Open a cursor to perform database operations
        cur = conn.cursor()

        # Execute the SQL query
        cur.execute(formatted_sql_query)

        # Fetch the result
        result = cur.fetchall()

        print(result)
        columns = [desc[0] for desc in cur.description]
        
        # Save results to a CSV file
        # csv_path = os.path.join(CSV_OUTPUT_DIR, 'csv_results.csv')
        with open(CSV_FILE_PATH, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns)  # Write headers
            writer.writerows(result)  # Write rows
            
        # Close communication with the database
            cur.close()
            conn.close()
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return None



def analyze_csv_with_llm(csv_path: str):
    try:
        # Read CSV content
        with open(csv_path, "r") as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
        
        # Prepare a prompt for the LLM
        header = ", ".join(rows[0])
        data_sample = "\n".join([", ".join(row) for row in rows[1:6]])  # Use a sample of first 5 rows
        prompt = (
            f"Here is a dataset with the following columns: {header}.\n"
            f"Sample data:\n{data_sample}\n\n"
            "Analyze this dataset and provide insights, trends, and actionable recommendations."
        )
        
        # Generate insights
        insights = llm(prompt)
        return insights
    except Exception as e:
        return f"An error occurred during analysis: {str(e)}"

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



with st.sidebar:
    # st.title("Models")
    model_name = st.selectbox(
        "Choose the Model?",
        ("GPT4"),
        # ("GPT4", "Llama3.3", "Mistral8b"),
    )


    # connection
    dbconnection = st.text_input("Enter the Database Connection URI")

    
    if st.button("Connect DB") and dbconnection:
        engine = create_engine(dbconnection)
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

# model = get_model(model_name)
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

st.title("InsightIQ")
st.caption("🚀 Unleash Private AI: LLMs Running Securely on Your Device - Powered by MistralAI and NVIDIA NIM")

# prompt=st.text_input("Enter Your Question From Doduments")


input_prompt=st.text_input("Enter Your Question From Documents")

# if input_prompt:
#     if "vectors" not in st.session_state:
#         vector_embedding()
import re
import pandas as pd
db_url = 'postgresql://postgres:admin@localhost:5432/olist'
engine = create_engine(db_url)


if input_prompt  and st.session_state.vectors is not None:
    # response = model.invoke(prompt)
    # st.write(model_name)
    # st.write(response.content)s

    # sql_query_prompt = f"""
    # Translate the following natural language query into an SQL query. If you cannot generate an SQL query, respond with 'No SQL query generated':
    # Query: {input_prompt}
    # """

    sql_query_prompt = f"""
    Based on the provided schema and table details, translate the following natural language query into an SQL query. Ensure the SQL query is as accurate as possible to answer the user’s request and provide explanation of the query's significance in general. If you cannot generate an SQL query, respond with 'No SQL query generated'.
    Natural Language Query: {input_prompt}
    """

    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    response=retrieval_chain.invoke({'input':input_prompt})

        # Write input_prompt and response['answer'] to queryresults.txt
    with open("queryresults.txt", "a") as file:
        file.write(f"Query: {input_prompt}\n")
        file.write(f"Answer: {response['answer']}\n")
        file.write("\n" + "-"*50 + "\n\n")
    
    st.write(response['answer'])


    sql_query_match = re.search(r"(SELECT.*?;)", response['answer'], re.DOTALL)
    if sql_query_match:
        sql_query = sql_query_match.group(1)
        print("Extracted SQL Query:")
        print(sql_query)
        if execute_query_tool(sql_query):
            print("Query executed successfully. Results saved to 'csv_results.csv'.")

            try:
                data = pd.read_csv(CSV_FILE_PATH)
                st.dataframe(data)

                insights = analyze_csv_with_llm(CSV_FILE_PATH)
                st.write(insights.content)

                plot_csv_data(CSV_FILE_PATH)
            except Exception as e:
                print( f"Error reading CSV file: {str(e)}")


        else:
            print("No SQL query found.")


