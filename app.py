import streamlit as st
from dotenv import load_dotenv


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
        pdf_docs = st.file_uploader(
            "Upload PDF data", accept_multiple_files=True)
        st.button("Process")


if __name__ == '__main__':
    main()