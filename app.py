import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import time
import streamlit.components.v1 as components

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

def handle_userinput(user_question):
    
    with st.session_state.container1:
        
        response = st.session_state.conversation({'question': user_question})
        print(response)
        st.session_state.chat_history = response['chat_history']

        st.write(user_template.replace(
                    "{{MSG}}", user_question), unsafe_allow_html=True)
        st.write(bot_template.replace(
                    "{{MSG}}", response['answer']), unsafe_allow_html=True)


def typewriter(text):
    components.html(
        f"""
        <div class="chat-message bot typewriter" >
            <div class="avatar">
                <img src="https://d2cbg94ubxgsnp.cloudfront.net/Pictures/2000x1125/9/9/3/512993_shutterstock_715962319converted_920340.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
            </div>
            <div class="message"></div>
        </div>
        <style>
        .chat-message {{
            padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
        }}
        .chat-message.user {{
            background-color: #2b313e
        }}
        .chat-message.bot {{
            background-color: #475063
        }}
        .chat-message .avatar {{
            width: 20%;
        }}
        .chat-message .avatar img {{
            max-width: 78px;
            max-height: 78px;
            border-radius: 50%;
            object-fit: cover;
        }}
        .chat-message .message {{
            width: 80%;
            padding: 0 1.5rem;
            color: #fff;
        }}
        
        
        </style>
        <script>
        const div = document.querySelector(".message");
        
        texto = "{text}"
        function effect(element, texto, i = 0) {{
            
            if (i === 0) {{
                element.textContent = "";
            }}

            element.textContent += texto[i];

            if (i === texto.length - 1) {{
                return;
            }}

            setTimeout(() => effect(element, texto, i+1), 40);
        }}

        effect(div, texto);
        </script>
        """,
        height=600,
    )


def main():

    load_dotenv()
    
    st.set_page_config(page_title="Chatbot for Asthma"
                       )
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
        typewriter("Welcome. ask anything about asthma.")
        
    
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