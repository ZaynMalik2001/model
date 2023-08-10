import streamlit as st
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
import os
import asyncio
import nest_asyncio

from dotenv import load_dotenv

# Allow nested event loops
nest_asyncio.apply()

# Load environment variables from .env file
load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
                     repo_id=repo_id,
                     model_kwargs={"temperature": 0.7, "max_new_tokens": 500})

template = """
You are a helpful AI assistant and provide the answer for the question asked politely.

{question}
"""

# Instantiate the chain
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

# Define an async function that calls the chain
async def call_chain(question):
    return await llm_chain.acall(question)  # Assuming 'cl' is previously defined

# Create a text input for user to enter their question
user_input = st.text_input("Please enter your question:")

if user_input:
    # Use an event loop to call the async function
    loop = asyncio.get_event_loop()
    res = loop.run_until_complete(call_chain(user_input))

    # "res" is a Dict. For this chain, we get the response by reading the "text" key.
    st.write(res["text"])
