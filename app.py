import os
import datetime
import json
import pandas as pd
import traceback
import PyPDF2
from dotenv import load_dotenv
import streamlit as st
from st_pages import Page, show_pages, add_page_title
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        st.session_state['conversation'] = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        st.session_state['chat_history'] = None

    st.set_page_config(
        page_title="Chat with your SQL Server database",
        page_icon="🧊"
    )

    #creating a title for the app
    st.title("Chat Application with local Database using the generative ai")

    uploaded_files = st.file_uploader("Upload CSV", type="csv", accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            file.seek(0)
        uploaded_data_read = [pd.read_csv(file, dtype=str) for file in uploaded_files]
        raw_data = pd.concat(uploaded_data_read)
        st.write(raw_data)
        # get the text chunks
        text_chunks = get_text_chunks(str(uploaded_data_read))
        # create vector store
        vectorstore = get_vectorstore(text_chunks)

        # create conversation chain
        st.session_state.conversation = get_conversation_chain(vectorstore)
    
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()



#pdf csv reader
def read_file(file):
    if file.name.endswith(".pdf"):
        try:
            pdf_reader=PyPDF2.PdfReader(file)
            text=""
            for page in pdf_reader.pages:
                text+=page.extract_text()
            return text
        except Exception as e:
            raise Exception("error reading the PDF file")
        
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    
    elif file.name.endswith(".csv"):
        return file.read().decode("utf-8")
    
    elif file.name.endswith(".rtf"):
        return file.read().decode("utf-8")
    
    elif file.name.endswith(".docx"):
        return file.read()
    
    elif file.name.endswith(".doc"):
        return file.read()
    
    #elif file.name.endswith(".jpeg"):
        #return read_text_from_jpeg_image(file)
    
    #elif file.name.endswith(".png"):
        #image = Image.open(file)  # Replace 'image.jpg' with the path to your image file
        # Use pytesseract to do OCR on the image
        #return pytesseract.image_to_string(image)
         

    
    else:
        raise Exception("unsopported file formate only pdf and text file supported")
