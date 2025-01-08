# import os

# from dotenv import load_dotenv
# load_dotenv() 

# nvidia_api_key  = os.getenv("NVIDIA_API_KEY")

# print(nvidia_api_key)


# from langchain_nvidia_ai_endpoints import ChatNVIDIA

# client = ChatNVIDIA(
#   model="mistralai/mistral-7b-instruct-v0.3",
# #   model ='mistralai/mixtral-8x7b-instruct-v0.1',
#   api_key="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", 
#   temperature=0.2,
#   top_p=0.7,
#   max_tokens=1024,
# )

# for chunk in client.stream([{"role":"user","content":"Write a limerick about the wonders of GPU computing."}]): 
#   print(chunk.content, end="")



# # OPENPI

# # from langchain_openai import ChatOpenAI

# # OPENAI_api_key  = os.getenv("OPENAI_API_KEY")

# # model = ChatOpenAI(model="gpt-4o-mini")

# # model.invoke("Hello, world!")



# # export LANGCHAIN_TRACING_V2="true"
# # export LANGCHAIN_API_KEY="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"

# from langchain_nvidia_ai_endpoints import ChatNVIDIA

# client = ChatNVIDIA(
#   model="mistralai/mixtral-8x7b-instruct-v0.1",
#   api_key="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", 
#   temperature=0.5,
#   top_p=1,
#   max_tokens=1024,
# )

# for chunk in client.stream([{"role":"user","content":"write a peom about keyboard."}]): 
#   print(chunk.content, end="")

# from openai import OpenAI

# client = OpenAI(
#   base_url = "https://integrate.api.nvidia.com/v1",
#   api_key = "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"
# )

# completion = client.chat.completions.create(
#   model="meta/llama-3.3-70b-instruct",
#   messages=[{"role":"user","content":"Write a code for binary tree."}],
#   temperature=0.5,
#   top_p=1,
#   max_tokens=1024,
#   stream=True
# )

# for chunk in completion:
#   if chunk.choices[0].delta.content is not None:
#     print(chunk.choices[0].delta.content, end="")

from langchain_nvidia_ai_endpoints import ChatNVIDIA

client = ChatNVIDIA(
  model="mistralai/mistral-7b-instruct-v0.3",
  api_key="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", 
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
)

for chunk in client.stream([{"role":"user","content":"Write a limerick about the wonders of GPU computing."}]): 
  print(chunk.content, end="")
