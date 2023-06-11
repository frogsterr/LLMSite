import os
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain import VectorDBQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain
from pydantic import BaseModel, Field
import sentence_transformers
from langchain.vectorstores import Chroma
from langchain.document_loaders import OnlinePDFLoader
import streamlit as st
from PyPDF2 import PdfReader
from langchain.agents.agent_toolkits import (create_vectorstore_agent, VectorStoreToolkit, VectorStoreInfo)





os.environ['OPENAI_API_KEY'] = 'Your Key Here!'
llm = OpenAI(model_name='text-davinci-003', temperature=.99)

'''
IGNORE
prompt = PromptTemplate(
    input_variables=["{level}, {keywords}"],
    template= "Build a series of questions that mimic an {level}-level exam "
             "based on the keywords/concepts given. Do not write anything but the questions."
             " At the end of each question, write the  ~ symbol."
             " Right after the question, write the answer. At the end of the answer"
             " write the * symbol. Keywords/Concepts: {keywords}"
)
'''

class Report(BaseModel):
    title: str = Field(description="title of the document")
    summary: str = Field(description="summary of the document with the key points")

parser = PydanticOutputParser(pydantic_object=Report)


# Converts PDF into document chunks

def pdfToChunks(file, chunk_size, chunk_overlap):

    loader = PdfReader(file)
    pages = loader.pages

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    report_list = [pages[page].extract_text() for page in range(len(pages))]
    texts = text_splitter.create_documents(report_list)

    return texts

# Use OpenAI embeddings and send vectors to ChromaDB

class vectorStore():
    def __init__(self, texts, llm):

        self.embeddings = OpenAIEmbeddings()
        self.vectordb = Chroma.from_documents(documents=texts, embedding=self.embeddings)

        self.vectorstore_info = VectorStoreInfo(
            name="Goldman Sachs Report",
            description="a banking annual report as a pdf",
            vectorstore = self.vectordb
        )

        self.toolkit = VectorStoreToolkit(vectorstore_info=self.vectorstore_info)

        # For Query style 1.
        self.agent_executor = create_vectorstore_agent(
            llm=llm,
            toolkit=self.toolkit,
            verbose=True
        )

        # For Query style 2. Question and Answer chain
        self.qachain = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=self.vectordb)







