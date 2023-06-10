import main
import streamlit as st
from PyPDF2 import PdfReader

# Title
st.title('sift.ai 🚀')

# File Uploader
user_pdf = st.file_uploader('Step 1. Upload your .pdf file', type="pdf")

if user_pdf is not None:
    document = main.pdfToChunks(user_pdf, chunk_size=100, chunk_overlap=20)
    st.write(document[0].page_content)



# User Prompt
prompt = st.text_input('Step 2. Ask away!')

if prompt:
    response = "why did you ask..."
    st.write(response)
