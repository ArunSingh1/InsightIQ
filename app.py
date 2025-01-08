from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import streamlit as st
import os
from dotenv import load_dotenv

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

## streamlit framework


# # openAI LLm 
# llm=ChatOpenAI(model="gpt-3.5-turbo")
# output_parser=StrOutputParser()
# chain=prompt|llm|output_parser

# if input_text:
#     st.write(chain.invoke({'question':input_text}))
GPT4 = ChatOpenAI(model_name="gpt-4", temperature=0)
Llama3_3 = ChatNVIDIA(
  model="meta/llama-3.3-70b-instruct"
#   api_key="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", 
#   temperature=0.5,
#   top_p=1,
#   max_tokens=1024,
)
# Mistral8b =ChatNVIDIA(
#   model="mistralai/mistral-7b-instruct-v0.3",
#   api_key="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", 
#   temperature=0.2,
#   top_p=0.7,
#   max_tokens=1024,
# )



def get_model(model_name):
    models = {
        "GPT4":GPT4,  # Replace with actual model initialization
        "Llama3.3": Llama3_3 ,  # Replace with actual model initialization
        "Mistral8b": Mistral8b  # Replace with actual model initialization
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
        ("GPT4", "Llama3.3", "Mistral8b"),
    )

model = get_model(model_name)

st.title("InsightIQ")
st.caption("🚀 Unleash Private AI: LLMs Running Securely on Your Device - Powered by MistralAI and NVIDIA NIM")



prompt=st.text_input("Enter Your Question From Doduments")


if prompt is not None:
    response = model.invoke(prompt)
    st.write(model_name)
    st.write(response.content)


