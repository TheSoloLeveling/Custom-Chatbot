import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template, js
from langchain.llms import HuggingFaceHub
import time
from streamlit.components.v1 import html

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

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore_openai(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_vectorstore_instructor(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(     #memory used by the chatbot stores messages in buffer - when called returns all messages stored
        memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(     #conversation chain 
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain

def typewriter(text: str, speed: int):
    tokens = text.split()
    container = st.empty()
    for index in range(len(tokens) + 1):
        curr_full_text = " ".join(tokens[:index])
        container.markdown(curr_full_text)
        time.sleep(1 / speed)

def handle_userinput(user_question):
    
    with st.session_state.container1:
        
        
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                #typewriter(text=user_template.replace("{{MSG}}", message.content), speed=10)
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True) 
        
        response = st.session_state.conversation({'question': user_question})
        print(response)
        st.session_state.chat_history = response['chat_history']

        st.write(user_template.replace(
                    "{{MSG}}", user_question), unsafe_allow_html=True)
        st.write(bot_template.replace(
                    "{{MSG}}", response), unsafe_allow_html=True)




def main():

    load_dotenv()
    
    st.set_page_config(page_title="Chatbot for Asthma",
                       page_icon=":activity:")
    st.header("Chatbot for Asthma")
    st.write(css, unsafe_allow_html=True)

    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "container1" not in st.session_state:
        st.session_state.container1 = None
  
    
    
    st.session_state.container1 = st.container()
    
    with st.session_state.container1:
        st.write(bot_template.replace(
                    "{{MSG}}", "ask anything about asthma ask anything about asthma ask anything about asthma ask anything about asthma"), unsafe_allow_html=True)
    
    html(js)
    user_question =  st.text_input("")

    if user_question:
        handle_userinput(user_question)
            

    with st.sidebar:
        st.subheader("Base Knowldge")
        docs = st.file_uploader(
            "Upload your data", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):

                #get pdf text
                raw_text = get_text_files(docs)
                

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # craete vector store, vector database containing embedding of each chunk of text
                vectorstore = get_vectorstore_openai(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()