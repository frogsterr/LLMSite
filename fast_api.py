from fastapi import FastAPI
from pydantic import BaseModel
from main import listToDoc, vectorStore, llm, template_prompt, response_parser


class UserInput(BaseModel):

    pdf_text: list
    subject: str
    problem_type: str
    class_type: str
    concepts: str


class DocumentRequest(BaseModel):
    answer: str


app = FastAPI()


@app.post("/model")
def operate(input: UserInput):
    doc = listToDoc(input.pdf_text, chunk_size=500, chunk_overlap=100)
    vectordb_agent = vectorStore(texts=doc, llm=llm)
    prompt = template_prompt(problem_type=input.problem_type, concepts=input.concepts,
                             class_type=input.class_type, subject=input.subject)
    response = vectordb_agent.agent_executor.run(prompt)
    updated_response = response_parser(response)
    return updated_response

