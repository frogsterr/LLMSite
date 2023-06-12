import os
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from PyPDF2 import PdfReader
from langchain.agents.agent_toolkits import (create_vectorstore_agent, VectorStoreToolkit, VectorStoreInfo)

os.environ['OPENAI_API_KEY'] = 'Your Key Here!'
llm = OpenAI(model_name='text-davinci-003', temperature=.7)

# Converts PDF into list of string pages
def pdfToList(file):

    loader = PdfReader(file)
    pages = loader.pages

    report_list = [pages[page].extract_text() for page in range(len(pages))]

    return report_list

# Splits lists into Documents

def listToDoc(lst, chunk_size, chunk_overlap):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    texts = text_splitter.create_documents(lst)

    return texts

# Converts pdf into Documents
def pdfToDoc(file, chunk_size, chunk_overlap):

    loader = PdfReader(file)
    pages = loader.pages

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    report_list = [pages[page].extract_text() for page in range(len(pages))]
    texts = text_splitter.create_documents(report_list)

    return texts


# Embeds Document and sends vectors to ChromaDB

class vectorStore():
    def __init__(self, texts, llm):

        self.embeddings = OpenAIEmbeddings()
        self.vectordb = Chroma.from_documents(documents=texts, embedding=self.embeddings)

        self.vectorstore_info = VectorStoreInfo(
            name="Document",
            description="Exam document",
            vectorstore =self.vectordb
        )

        self.toolkit = VectorStoreToolkit(vectorstore_info=self.vectorstore_info)

        # Query style 1.
        self.agent_executor = create_vectorstore_agent(
            llm=llm,
            toolkit=self.toolkit,
            verbose=True
        )

# Template maker for LLM
def template_prompt(problem_type, class_type, concepts, subject):

    prompt = f"Generate multiple {problem_type}" \
              f"questions that you would find in a standard {class_type}-level class using the {subject} document provided." \
             f" Use the Document tool. Questions should be related to the following concepts: {concepts}. " \
              f"After writing the end of each question, write the ~ symbol. After the ~ symbol," \
              f"write the answer. At the end of the answer, write the * symbol. RETURN QUESTIONS AND ANSWERS. FOLLOW" \
              f"INSTRUCTIONS"

    return prompt



# Parses LLM responses into questions and answers
def response_parser(response):

    question_bank = {}

    message = response.split('*')
    message = [x.replace("\n", "") for x in message]

    for q in range(len(message) - 1):
        qna = message[q].split('~')
        question_bank[q] = {'question': f'{qna[0]}', 'answer': f'{qna[1]}'}

    return question_bank

