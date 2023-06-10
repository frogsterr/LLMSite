import main
import streamlit as st
from PyPDF2 import PdfReader

# Title
st.title('sift.ai 🚀')

# File Uploader
user_pdf = st.file_uploader('Step 1. Upload your .pdf file', type="pdf")

if user_pdf is not None:
    pdf_reader = PdfReader(user_pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    print(text)



# User Prompt
prompt = st.text_input('Step 2. Ask away!')

if prompt:
    response = "why did you ask..."
    st.write(response)
