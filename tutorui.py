import requests
import json
import streamlit as st
from main import pdfToList

st.title("Sift.ai x StepUp | A New Way to Learn")

# User input
user_pdf = st.file_uploader('Upload your .pdf file', type="pdf")
if user_pdf is not None:
    pages_list = pdfToList(user_pdf)
    subject = st.selectbox('Subject Type', ('Math', 'History', 'Science', 'English', 'Business'))
    problem_type = st.selectbox('Problem Type', ('Multiple Choice', 'Short Answer'))
    class_type = st.selectbox('Class Level', ('AP', 'IB', 'College'))
    concept = st.text_input('What keywords')

    # Converting the inputs into json format
    user_input = {"pdf_text": pages_list, "subject": subject, "problem_type": problem_type,
                              "class_type": class_type, "concepts": concept, "answer": ""}

    if st.button('Go'):
        try:
            # Fetch QNA Bank from Model API
            res = requests.post(url="http://127.0.0.1:8000/model", data=json.dumps(user_input))
            qna_bank = res.json()

            # Parse QNA Bank into Question Fields
            for qna in range(len(qna_bank)):
                st.write(qna_bank[f"{qna}"]["question"])
                user_answer = st.text_input("Write Here", key=qna)

                # Checks if user answer is equal to QnA answer (Method should be different for short answer response)
                if user_answer == qna_bank[f"{qna}"]["answer"]:
                    st.write("Correct!")

                # Writes QNA Bank Answer, fetches similarity report from answer report API
                    with st.expander('Answer'):
                        st.write(f'{qna_bank[f"{qna}"]["answer"]}\n\n')
                        user_input["answer"] = qna_bank[f"{qna}"]["answer"]
                        answer_doc = requests.post(url="http://127.0.0.1:8000/answer_report",
                                                   data=json.dumps(user_input))
                        st.write(answer_doc)

        except ValueError:
            st.write(user_input)
            st.write("Error, please refresh page and try again")

