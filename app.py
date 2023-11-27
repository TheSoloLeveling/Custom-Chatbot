import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

#def get_images
#def get_file_text
def get_text_files(text_files):
    text = ""
    for doc in text_files:
        file_contents = doc.getvalue().decode("utf-8")
        text += file_contents
        text = text + ' '
    return text
   

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def main():

    load_dotenv()
    st.set_page_config(page_title="Chatbot for Asthma",
                       page_icon=":activity:")
    st.header("Chatbot for Asthma")
    #st.write(css, unsafe_allow_html=True)

    user_question = st.text_input("Ask anything about asthma:")
    if user_question:
        print("test")
        #handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Base Knowldge")
        docs = st.file_uploader(
            "Upload your data", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                #get pdf text
                raw_text = get_text_files(docs)
                st.write(raw_text)
                # get the text chunks

                # craete vector store


if __name__ == '__main__':
    main()