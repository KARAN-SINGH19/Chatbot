from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

import chainlit as cl
id="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"

repo_id = "tiiuae/falcon-7b-instruct"

llm = HuggingFaceHub(
    huggingfacehub_api_token=id,
    repo_id=repo_id,
    model_kwargs={"temperature": 0.7, "max_new_tokens": 500}
)


template = """Question: {question}

Answer: Let's think step by step."""


@cl.on_chat_start
def main():
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: str):
    llm_chain = cl.user_session.get("llm_chain")  
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(content=res["text"]).send()
