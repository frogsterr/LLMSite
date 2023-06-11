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
    if subject:
        problem_type = st.selectbox('Problem Type', ('Multiple Choice', 'Short Answer'))
        if problem_type:
            class_type = st.selectbox('Class Level', ('AP', 'IB', 'College'))
            if class_type:
                concept = st.text_input('What keywords')

                # Converting the inputs into json format
                user_input = {"pdf_text": pages_list, "subject": subject, "problem_type": problem_type,
                                  "class_type": class_type, "concepts": concept}

                # Fetch Model API
                if st.button('Go'):
                    res = requests.post(url="http://127.0.0.1:8000/model", data=json.dumps(user_input))
                    qna_bank = res.json()

                    for qna in range(len(qna_bank)):
                        st.write(qna_bank[f"{qna}"]["question"])
                        user_answer = st.text_input("Write Here", key=qna)
                        with st.expander('Answer'):
                            st.write(qna_bank[f"{qna}"]["answer"])



