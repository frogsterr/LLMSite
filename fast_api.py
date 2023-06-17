from fastapi import FastAPI
from pydantic import BaseModel
from main import listToDoc, vectorStore, llm, template_prompt, response_parser


# Pydantic Object w/ User Input
class UserInput(BaseModel):

    pdf_text: list
    subject: str
    problem_type: str
    class_type: str
    concepts: str
    answer: str


app = FastAPI()

# Returns Parsed LLM Agent Response
@app.post("/model")
def operate(input: UserInput):
    # Convert Pdf-list to Document Objects
    doc = listToDoc(input.pdf_text, chunk_size=500, chunk_overlap=100)
    # Instantiate LangChain Object
    vectordb_agent = vectorStore(texts=doc, llm=llm)
    # Create prompt based on User Input from Pydantic Model
    prompt = template_prompt(problem_type=input.problem_type, concepts=input.concepts,
                             class_type=input.class_type, subject=input.subject)
    # Response from LLM / LangChain agent
    response = vectordb_agent.agent_executor.run(prompt)
    # Parse response into QNA bank
    updated_response = response_parser(response)
    return updated_response

# Finds section of article related to answer
@app.post("/answer_report")
def similarity(input: UserInput):
    # Convert Pdf-list to Document Objects
    doc = listToDoc(input.pdf_text, chunk_size=500, chunk_overlap=100)
    # Instantiate LangChain Object
    vectordb_agent = vectorStore(texts=doc, llm=llm)
    # Finds similar embedded section of article
    search = vectordb_agent.vectordb.similarity_search_with_score(input.answer)
    return search[0][0].page_content
