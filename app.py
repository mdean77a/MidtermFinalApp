# Prototype for Hugging Face upload 
import os
# import logging
# logging.getLogger("transformers").setLevel(logging.INFO)
# logging.getLogger("pytorch").setLevel(logging.INFO)
# logging.getLogger("sentence_transformers").setLevel(logging.INFO)

from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
import chainlit as cl
from dotenv import load_dotenv
import loadReferenceDocuments
import prompts
import splitAndVectorize
import defaults
from langchain_huggingface import HuggingFaceEmbeddings

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Set to 'false' to disable parallelism

load_dotenv()

OPENAI_API_KEY = os.getenv("NEWKEY")

# Path to my directory containing PDF files
directory = "References/"

separate_pages, one_document = loadReferenceDocuments.loadReferenceDocuments("References/")
finetune_embeddings = None 
def get_embedding_model():
    global finetune_embeddings
    if finetune_embeddings is None:
        finetune_embeddings = HuggingFaceEmbeddings(model_name="Mdean77/finetuned_arctic")
    return finetune_embeddings

finetune_embeddings = get_embedding_model()

chunk_split_vectorstore = splitAndVectorize.createVectorstore(
    one_document,
    "chunk_split_collection",
    chunk_size=800,
    chunk_overlap=400,
    embedding_model=finetune_embeddings
)

chunk_split_retriever = chunk_split_vectorstore.as_retriever()

llm = defaults.default_llm

rag_prompt = ChatPromptTemplate.from_template(prompts.rag_prompt_template)

rag_chain = (
    {"context": itemgetter("question") | chunk_split_retriever, "question": itemgetter("question")}
    | rag_prompt | llm | StrOutputParser()
)

# This code is executed outside of the ChainLit environment to prove the chain works
# The output is shown in terminal - I can delete this code if necessary
page_response = (rag_chain.invoke({"question": "List the ten major risks of AI?"}))
print(page_response)


@cl.on_chat_start
async def on_chat_start():
    welcome_message = """
#  Responsible AI Development Q&A Chatbot!

As you know, AI is a hot topic, and our company is engaged in developing software applications.  We believe that we will inevitably be developing software that utilizes AI, but people are rightfully concerned about the implications of using AI.  Nobody seems to understand the right way to think about building ethical and useful AI applications for enterprises.  

I am here to help you understand how the AI industry is evolving, especially as it relates to politics. Many people believe that the best guidance is likely to come from the government, so I have thoroughly read two important documents:

1.  White House Blueprint for an AI Bill of Rights, Making Automated Systems Work for the American People, October 2022
2.  Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile (NIST AI 600-1)

Ask me questions about AI, its implications, where the industry is heading, etc.  I will try to help!

"""

    await cl.Message(content=welcome_message).send()

    # Set the chain in the session
    cl.user_session.set("chain", rag_chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    result =  chain.invoke({"question":message.content})
    await cl.Message(content=result).send()
    # msg = cl.Message(content=result)
    # await msg.send()

