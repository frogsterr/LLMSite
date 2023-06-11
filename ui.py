import main
import streamlit as st
from PyPDF2 import PdfReader

# Title
st.title('sift.ai 🚀')


# File type
file_type = st.selectbox('Step 1. Select File Type', ('PDF', 'CSV', 'Text'))

if file_type == 'PDF':
    # Upload
    user_pdf = st.file_uploader('Step 2. Upload your .pdf file', type="pdf")

    if user_pdf is not None:
        document = main.pdfToChunks(user_pdf, chunk_size=1000, chunk_overlap=200)
        vectordb_agent = main.vectorStore(texts=document, llm=main.llm)


# User Prompt
        prompt = st.text_input('Step 3. Ask away!')

        if prompt:
            try:
                # Query Style 1.

                response = vectordb_agent.agent_executor.run(prompt)

                # Query Style 2.

                # response = vectordb_agent.qachain.run(prompt) 

                st.write(response)
            except NameError:
                st.write("Please upload file")

            with st.expander('Document Similarity Search'):
                # Find the relevant pages
                search = vectordb_agent.vectordb.similarity_search_with_score(prompt)
                # Write out the first
                st.write(search[0][0].page_content)

