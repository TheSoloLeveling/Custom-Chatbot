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



def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents


def split_docs(documents,chunk_size=500,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs





############################################################################################################################################
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

def get_conversation_chain(vectorstore, text_chunks, topCount):

    # Create the Transform
    vectorizer = TfidfVectorizer()

    top_keywords_list = []
    for text in text_chunks:
        # Tokenize and build vocab
        vectorizer.fit([text])

        # Encode document
        vector = vectorizer.transform([text])

        # Summarize encoded vector
        keywords = vectorizer.get_feature_names_out()

        # Sort the TF-IDF scores in descending order
        sorted_indices = np.argsort(vector.toarray()).flatten()[::-1]

        # Get the top 3 keywords
        top_keywords = [keywords[i] for i in sorted_indices[:topCount]]
        top_keywords_list.append(top_keywords)

        keywords_string = ', '.join([', '.join(sublist) for sublist in top_keywords_list])

    
    general_system_template = r""" 
    the given context {context}:
    ----
    
    answer only if its related to most of these key words :
    """ + keywords_string + r"""
    ----
    the history of the your chat with the human :

    {chat_history}
    ----
    """
    
    general_user_template = "Question:```{question},```"
    messages = [
                SystemMessagePromptTemplate.from_template(general_system_template),
                HumanMessagePromptTemplate.from_template(general_user_template)
    ]
    # Create the prompt template
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferWindowMemory( k=6,    #memory used by the chatbot stores messages in buffer - when called returns all messages stored
        memory_key='chat_history', return_messages=True)

    st.session_state.chatMemory = memory

    
    conversation_chain = ConversationalRetrievalChain.from_llm(     #conversation chain 
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=st.session_state.chatMemory,
        get_chat_history=lambda h:h,
        combine_docs_chain_kwargs={'prompt': qa_prompt}
    )

    return conversation_chain

def handle_userinput(user_question):
    st.session_state.status = False
    with st.session_state.container1:
        
        dummy = copy.deepcopy(st.session_state.chatMemory)
        print("dummy : " + str(dummy))
        response = st.session_state.conversation({'question': user_question})
        #print(response)

        #score = sim(str(user_question), str(response['answer']))
        #print("similarity between " + str(user_question) + " and " + str(response['answer']) + " is : " + str(score))
        st.session_state.chatMemory = response['chat_history']
        st.session_state.displayMemory.append(user_question)
        st.session_state.displayMemory.append(response['answer'])
        
        #if score > 0.1 :                                                                                # add other scores between the input and memory
        #    #st.session_state.chat_history = response['chat_history']
        #    st.session_state.chatMemory = response['chat_history']
        #    st.session_state.displayMemory.append(user_question)
        #    st.session_state.displayMemory.append(response['answer'])
        #else:
            #st.session_state.chat_history = response['chat_history']
        #    st.session_state.chatMemory = dummy
        #    print("after update memory : " + str(st.session_state.chatMemory))
        #   st.session_state.displayMemory.append(user_question)
         #   st.session_state.displayMemory.append("I am sorry i can't respond to that, not on my personal knwoldge, i need more details !!!")
       
        #print(type( st.session_state.chat_history))
        #typewriterUser(user_question)
        #typewriterBot(response['answer'])
        #st.session_state.status = True

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
    if "displayMemory" not in st.session_state:
        st.session_state.displayMemory = []
    
    if "chatMemory" not in st.session_state:
        st.session_state.chatMemory = None

    if "fixedMemory" not in st.session_state:
        st.session_state.fixedMemory = None
    
  
    #st.session_state.status = True
    left, d1, d2, right = st.columns((8,1,1,3))
    
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
        st.session_state.status = True
        handle_userinput(st.session_state.user_question)

    #Print In Screen AFTER all UPDATEs
    with st.session_state.container1:
        with left:
            result = typewriterBot("Welcome Back. ask anything about asthma.")
            #st.write("result = ", result)
    
    print(st.session_state.status)
    #print(st.session_state.displayMemory)
    print(st.session_state.chatMemory)
    if st.session_state.displayMemory:
        with st.session_state.container1:
            with left:
                for i, message in enumerate(st.session_state.displayMemory):
                    if i % 2 == 0:
                        typewriterUser(message)
                        #st.write(user_template.replace(
                            #   "{{MSG}}", message.content), unsafe_allow_html=True)
                    else:
                        typewriterBot(message)

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
        
        documents = load_docs(r'C:\Users\bouzi\OneDrive\Phd\Janvier 2024\data')
        st.write(len(documents))
        docs = split_docs(documents)
        for i in range(0, len(docs)+1):
            st.write(i)
            st.write(docs[i].page_content)


        st.subheader("Created models")
        with st.container(border=True):
                
                st.write("model 1")
                st.divider()
                st.write("model 2")
                st.divider()
                st.write("model 3")
          
        vectorstore = None
        #st.markdown("""---""")
        #st.text_area("test")
        st.markdown("""<hr style="height:4px;border:none;color:#0080ff;background-color:#0080ff;" /> """, unsafe_allow_html=True)
        docs = st.file_uploader(
            "Upload Base Knowldge", accept_multiple_files=True)
        if st.button("Process"):
            
            with st.spinner("Processing"):

                #get pdf text
                raw_text = get_text_files(docs)
                

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # craete vector store, vector database containing embedding of each chunk of text
                vectorstore = get_vectorstore_openai(text_chunks)
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore, text_chunks, 6)
        #st.markdown("""---""")
            





from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import nltk
nltk.download('stopwords')

def sim(X,Y):
    
    # X = input("Enter first string: ").lower() 
    # Y = input("Enter second string: ").lower() 
    # tokenization 
    X_list = word_tokenize(X)  
    Y_list = word_tokenize(Y) 
    
    # sw contains the list of stopwords 
    sw = stopwords.words('english')  
    l1 =[];l2 =[] 
    
    # remove stop words from the string 
    X_set = {w for w in X_list if not w in sw}  
    Y_set = {w for w in Y_list if not w in sw} 
    
    # form a set containing keywords of both strings  
    rvector = X_set.union(Y_set)  
    for w in rvector: 
        if w in X_set: l1.append(1) # create a vector 
        else: l1.append(0) 
        if w in Y_set: l2.append(1) 
        else: l2.append(0) 
    c = 0
    
    # cosine formula  
    for i in range(len(rvector)): 
            c+= l1[i]*l2[i] 
    cosine = c / float((sum(l1)*sum(l2))**0.5) 
    print("similarity: ", cosine) 

    return cosine

if __name__ == '__main__':
    main()