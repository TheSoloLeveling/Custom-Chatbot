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
    st.session_state.status = False
    with st.session_state.container1:
        response = st.session_state.conversation({'question': user_question})
        print(response)
        st.session_state.chat_history = response['chat_history']

        
        
        #print(type( st.session_state.chat_history))
        #typewriterUser(user_question)
        #typewriterBot(response['answer'])
        #st.session_state.status = True

def typewriterUser(text):
    # Estimate the number of lines
    average_chars_per_line = 64  # This is an estimate; adjust based on your actual content and styling
    line_height = 1.5  # Adjust based on your styling
    font_size = 16  # Font size in pixels; adjust as needed

    # Calculate the number of lines the text will occupy
    num_lines = len(text) / average_chars_per_line
    estimated_height = 0
    # Calculate the estimated height in pixels
    if num_lines > 3 :
        estimated_height = num_lines * line_height * font_size
    else:
        estimated_height = 90
    components.html(
        f"""
        <div class="chat-container">
            <div class="chat-message bot typewriter">
                <div class="avatar">
                    <img src="https://th.bing.com/th/id/OIP.AgSYsO763QByj6ib0orqNgHaHa?rs=1&pid=ImgDetMain" style="max-height: 40px; max-width: 200px; border-radius: 50%; object-fit: cover;">
                </div>    
                <div class="message">{text}</div>
            </div>
        </div>
        <style>
        /* Set up the grid container */
        .chat-container {{
            display: grid;
            grid-template-columns: auto; /* One column layout */
            grid-gap: 10px; /* Adjust the gap between grid items */
            margin: 0 auto; /* Center the container */
            
        }}

        /* Style for each chat message */
        .chat-message {{
            display: flex;
            align-items: center;
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #475063;
            color: #fff;
            margin: 0; /* Remove any default margin */
            align-items: flex-start;
        }}

        /* Style for the avatar within the message */
        .chat-message .avatar {{
            width: 20%;
            padding-right: 0.5rem; /* Space between avatar and message */
            align-self: flex-start;
        }}

        /* Style for the message text */
        .chat-message .message {{
            width: 80%;
            white-space: pre-wrap;
            word-wrap: break-word;
            flex-grow: 1;
        }} 
        </style>
        """,
        height=int(estimated_height),
        
    )

def typewriterBot(text):
    # Estimate the number of lines
    average_chars_per_line = 64  # This is an estimate; adjust based on your actual content and styling
    line_height = 1.5  # Adjust based on your styling
    font_size = 16  # Font size in pixels; adjust as needed

    # Calculate the number of lines the text will occupy
    num_lines = len(text) / average_chars_per_line
    estimated_height = 0
    # Calculate the estimated height in pixels
    if num_lines > 3 :
        estimated_height = num_lines * line_height * font_size
    else:
        estimated_height = 90
    components.html(
        f"""
        <div class="chat-container">
            <div class="chat-message bot typewriter" >
                <div class="avatar">
                    <img src="https://d2cbg94ubxgsnp.cloudfront.net/Pictures/2000x1125/9/9/3/512993_shutterstock_715962319converted_920340.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
                </div>
                <div class="message"></div>
            </div>
        </div>
        <style>
        /* Set up the grid container */
        .chat-container {{
            display: grid;
            grid-template-columns: auto; /* One column layout */
            grid-gap: 10px; /* Adjust the gap between grid items */
            margin: 0 auto; /* Center the container */
            
        }}

        /* Style for each chat message */
        .chat-message {{
            display: flex;
            align-items: center;
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #475063;
            color: #fff;
            margin: 0; /* Remove any default margin */
            align-items: flex-start;
        }}

        /* Style for the avatar within the message */
        .chat-message .avatar {{
            width: 20%;
            padding-right: 0.5rem; /* Space between avatar and message */
            align-self: flex-start;
        }}

        /* Style for the message text */
        .chat-message .message {{
            width: 80%;
            white-space: pre-wrap;
            word-wrap: break-word;
            flex-grow: 1;
        }} 
        </style>
        <script>
        const div = document.querySelector(".message");
        
        const texto = `{text}`;
        function effect(element, texto, i = 0) {{
            
            if (i === 0) {{
                element.textContent = "";
            }}

            element.textContent += texto[i];

            if (i === texto.length - 1) {{
                return;
            }}

            setTimeout(() => effect(element, texto, i+1), 25);
        }}

        effect(div, texto);
        </script>
        """,
        height=int(estimated_height),
        
    )


def main():

    load_dotenv()
    
    st.set_page_config(page_title="Chatbot for Asthma"
                       )
    st.header("Chatbot for Asthma")
    st.write(css, unsafe_allow_html=True)

    if "status" not in st.session_state:
        st.session_state.status = True
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "container1" not in st.session_state:
        st.session_state.container1 = None
    if "user_question" not in st.session_state:
        st.session_state.user_question = None
  
    #st.session_state.status = True
    st.session_state.container1 = st.container()
    toggle_all= True

    st.session_state.user_question = st.text_input("")

    if st.session_state.user_question:
        st.session_state.status = True
        handle_userinput(st.session_state.user_question)

    #Print In Screen AFTER all UPDATEs
    with st.session_state.container1:
        typewriterBot("Welcome Back. ask anything about asthma.")
    
    print(st.session_state.status)
    if st.session_state.chat_history:
        with st.session_state.container1:
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    typewriterUser(message.content)
                    #st.write(user_template.replace(
                        #   "{{MSG}}", message.content), unsafe_allow_html=True)
                else:
                    typewriterBot(message.content)
        
    
    
    if toggle_all:
        #This looks for any input box and applies the code to it to stop default behavior when focus is lost
        components.html(
            """
        <script>
        const doc = window.parent.document;
        const inputs = doc.querySelectorAll('input');
        inputs.forEach(input => {
            // Make each input required
            input.required = true;
        });

        inputs.forEach(input => {
        input.addEventListener('focusout', function(event) {
            event.stopPropagation();
            event.preventDefault();
            console.log("lost focus")
        });
        });

        </script>""",
            height=0,
            width=0,
        )

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