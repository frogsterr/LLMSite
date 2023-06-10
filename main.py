from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain
from pydantic import BaseModel, Field
import sentence_transformers
from langchain.vectorstores import Chroma
from langchain.document_loaders import OnlinePDFLoader
import streamlit as st



#llm = OpenAI(model_name='text-davinci-003', temperature=.9)
#chain = LLMChain(llm=llm, prompt=prompt)

prompt = PromptTemplate(
    input_variables =["user_prompt"],
    template="{user_prompt}"
)

class Report(BaseModel):
    title: str = Field(description="title of the document")
    summary: str = Field(description="summary of the document with the key points")

parser = PydanticOutputParser(pydantic_object=Report)


# Converts PDF into document chunks

def pdfToChunks(file, chunk_size, chunk_overlap):

    loader = PyPDFLoader(file)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    report_list = [pages[page].page_content for page in range(len(pages))]
    texts = text_splitter.create_documents(report_list)

    return texts

# Use OpenAI embeddings and send vectors to ChromaDB

def vectorStore(texts):

    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_texts(texts, embeddings)

print(pdfToChunks('example.pdf', 200,20))




