import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder
)
import time
import streamlit.components.v1 as components
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import copy
import io
import numpy as np



############################################################################################################################################

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone 
import openai
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationChain
from langchain.llms import CTransformers
from langchain.embeddings import SentenceTransformerEmbeddings
from accelerate import Accelerator
from openai import OpenAI
import os


def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents


def split_docs(documents,chunk_size=700,chunk_overlap=60):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs


def get_vectorstore(docs):

    embeddings = OpenAIEmbeddings()
    #embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
    
    pinecone.init(
    api_key="af1d2797-f231-4231-a0d8-8f230b4674eb",  # find at app.pinecone.io
    environment="gcp-starter"  # next to api key in console
    )

    index = Pinecone.from_documents(docs, embeddings, index_name='medical-chatbot')



def get_similiar_docs(index, query, k=1, score=False):
  
  if score:
    similar_docs = index.similarity_search_with_score(query,k=k)
  else:
    similar_docs = index.similarity_search(query,k=k)
  return similar_docs


def get_local_llm(local_llm):


    config = {
        'max_new_tokens': 1024,
        'context_length': 8000,
        'repetition_penalty': 1.1,
        'temperature': 0.1,
        'top_k': 50,
        'top_p': 0.9,
        'stream': True,
    }

    llm = CTransformers(
        model=local_llm,
        model_type="llama",
        gpu_layers=50,
        **config
    )
    
    return llm
    
def get_conversation_chain():

    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    #llm = get_local_llm("meditron-7b.Q4_K_M.gguf")
    memory = ConversationBufferWindowMemory(k=3, memory_key='history', return_messages=True)
    st.session_state.chatMemory = memory

    system_msg_template = SystemMessagePromptTemplate.from_template(template="""you are a medical assistant that help patient in morroco, Answer the question , in french standart language, as truthfully as possible using the provided context and don't add informations by yourself answer only from the provided context, 
    and if the answer is not contained within the text below, say 'I don't know', and if the answer is not related to morroco, say 'i don't know '""")

    human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

    prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])
    #prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history")])

    conversation = ConversationChain(memory=st.session_state.chatMemory, prompt=prompt_template, llm=llm, verbose=True)

    return conversation

def query_refiner(conversation, query):
    client = OpenAI()
    prompt = f"Given the following user query and conversation log, formulate a question that would be asked by patient holding conversation with a doctor or medical assistant, and that would be the most relevant to provide an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"
    response = client.completions.create(
        prompt=prompt,
        model="gpt-3.5-turbo-instruct",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text


def find_match(input):
    embeddings = OpenAIEmbeddings()
    #embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
    input_em = embeddings.embed_query(input)
    
    result = st.session_state.index.query(input_em, top_k=4, includeMetadata=True)
    
    # Check if there are at least two matches
    if len(result['matches']) >= 4:
        return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text'] + "\n" + result['matches'][2]['metadata']['text'] + "\n" + result['matches'][3]['metadata']['text']
    if len(result['matches']) >= 3:
        return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text'] + "\n" + result['matches'][2]['metadata']['text']
    if len(result['matches']) >= 2:
        return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text'] 
    elif len(result['matches']) == 1:
        return result['matches'][0]['metadata']['text']
    else:
        return ""

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string


def handle_userinput(user_question):
    st.session_state.status = False
    with st.session_state.container1:

        conversation_string = get_conversation_chain()
        refined_query = query_refiner(conversation_string, user_question)
        #st.subheader("extended query:")
        #st.write(refined_query)
        context = find_match(refined_query)

        response = st.session_state.conversation.predict(input=f"Context:\n {context} \n\nsi c'est pas une question, repond normal, si c'est une question repond uniquement a partir du context, si il n'y a pas dit je ne sais pas:\n{user_question}")
        #response = st.session_state.conversation.predict(user_question)
        
        st.session_state.requests.append(user_question)
        st.session_state.responses.append(response) 
    


############################################################################################################################################
        

def typewriterUser(text):
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
        body {{
            background-color: #262626;
        }}
        /* Set up the grid container */
        .chat-container {{
            
        }}

        /* Style for each chat message */
        .chat-message {{
            display: flex;
            align-items: center;
            padding: 1rem;
            border-radius: 0.5rem;
            background: linear-gradient(to left, #004080,  #0080ff);
            color: #fff;
            margin: 0; /* Remove any default margin */
            align-items: flex-start;
            border: 0.02rem solid;
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
            font-size: 18px;
        }} 
        </style>
        """,
        height = 130, scrolling = True
        
    )

def typewriterBot(text):
    
    components.html(
        f"""
        <div class="chat-container">
            <div class="chat-message bot typewriter" >
                <div class="avatar">
                    <img src="https://d2cbg94ubxgsnp.cloudfront.net/Pictures/2000x1125/9/9/3/512993_shutterstock_715962319converted_920340.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
                </div>
                <div class="message">{text}</div>
            </div>
        </div>
        <style>
        html, body {{
            background-color: #262626;
            
        }}
        /* Set up the grid container */
        .chat-container {{
        }}

        /* Style for each chat message */
        .chat-message {{
            display: flex;
            align-items: center;
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #475062;
            color: #fff;
            margin: 0; /* Remove any default margin */
            align-items: flex-start;
            border: 0.02rem solid;
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
            font-size: 18px;
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

            setTimeout(() => effect(element, texto, i+1), 5);
        }}
        effect(div, texto);

        const dataToSend = "test dataTotSend !!!!!!!!"
        Streamlit.setComponentValue(3.14);
        </script>
        """, height = 130, scrolling = True
    )
    
    #data_from_js = st.session_state.get('data_from_js', None)

    #return data_from_js


def main():

    load_dotenv()
    st.set_page_config(page_title="Chatbot for Asthma",
                       layout='wide')
    
    st.header("Chatbot for Asthma", divider="blue")
    st.write(css, unsafe_allow_html=True)

    
    if "valid" not in st.session_state:
        st.session_state.valid = True
    if "chatMemory" not in st.session_state:
        st.session_state.chatMemory = None
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "container1" not in st.session_state:
        st.session_state.container1 = None
    if "user_question" not in st.session_state:
        st.session_state.user_question = None
    if 'responses' not in st.session_state:
        st.session_state['responses'] = ["Comment je peux t'aider ?"]
    if 'requests' not in st.session_state:
        st.session_state['requests'] = []
    if "index" not in st.session_state:
        st.session_state.index = None


    pinecone.init(
    api_key="af1d2797-f231-4231-a0d8-8f230b4674eb",  # find at app.pinecone.io
    environment="gcp-starter"  # next to api key in console
    )

    if  st.session_state.valid:
        st.session_state.valid = False
        existing_indexes = pinecone.list_indexes()
        if "medical-chatbot" in existing_indexes:
            print(f"Index already exists.")
            pinecone.delete_index('medical-chatbot')
        
        pinecone.create_index(name='medical-chatbot', dimension=1536, metric="cosine")
        st.session_state.index = pinecone.Index("medical-chatbot")
        print(f"Index does not exist.")
    
    #st.session_state.status = True
    left, d1, d2, right = st.columns((13,1,1,4))
    
    st.session_state.container1 = st.container()
    toggle_all= True

    with st.session_state.container1:
            with left:
                a, b = st.columns((1,12))
                with a:
                    st.markdown('<button class="primaryButton" data-tooltip="New Conversation"> + </button>', unsafe_allow_html=True)
                with b:
                    st.session_state.user_question = st.text_input("", placeholder="ask a question...")

    if st.session_state.user_question:
        handle_userinput(st.session_state.user_question)

    
    with st.session_state.container1:
        with left:
            if st.session_state['responses']:
                for i in range(len(st.session_state['responses'])):
                    typewriterBot(st.session_state['responses'][i])
                    if i < len(st.session_state['requests']):
                        typewriterUser(st.session_state["requests"][i])
                    

    with st.session_state.container1:
        right.subheader("Recent activity")
        with right:
           with st.container(border=True):
                st.button("conversation 1", key ="1")
                st.divider()
                st.button("conversation 2", key ="2")
                st.divider()
                st.button("conversation 3", key="3")
    
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
        #st.subheader("Created models")
        #with st.container(border=True):
                
        #        st.write("model 1")
        #        st.divider()
        #        st.write("model 2")
        #        st.divider()
        #        st.write("model 3")
          
        st.markdown("""<hr style="height:4px;border:none;color:#0080ff;background-color:#0080ff;" /> """, unsafe_allow_html=True)
        docs = st.file_uploader(
            "Upload Base Knowldge", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):

                random_file_name = uuid.uuid4().hex
                directory = r'C:\Users\bouzi\OneDrive\Phd\Janvier 2024'
                file_path = os.path.join(directory, random_file_name)

                for doc in docs:
                    save_uploaded_file(doc, file_path)

                documents = load_docs(file_path)
                sDocs = split_docs(documents)
                #st.write(sDocs)
                

                get_vectorstore(sDocs)

                #query = "How is India economy"
                #similar_docs = get_similiar_docs(index=st.session_state.index, query=query, score=True)
                #st.write(similar_docs)

                #result = st.session_state.index.query(input_em, top_k=2, includeMetadata=True)

                st.session_state.conversation = get_conversation_chain()



                        

            
import os 
import uuid

def save_uploaded_file(uploaded_file, directory):
    if uploaded_file is not None:
        # Make sure the directory exists, create it if it doesn't
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save the file to the specified directory
        with open(os.path.join(directory, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    return False


if __name__ == '__main__':
    main()