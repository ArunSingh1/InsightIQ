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

import os
os.makedirs('./data', exist_ok=True)


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
    st.title("Models")
    model_name = st.selectbox(
        "Choose the Model?",
        ("GPT4"),
        # ("GPT4", "Llama3.3", "Mistral8b"),
    )


    st.session_state["upload_status"] = False
    uploaded_files = st.sidebar.file_uploader("Upload Files", accept_multiple_files=True)
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

if input_prompt  and st.session_state.vectors is not None:
    # response = model.invoke(prompt)
    # st.write(model_name)
    # st.write(response.content)s

    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    response=retrieval_chain.invoke({'input':input_prompt})
    st.write(response['answer'])


