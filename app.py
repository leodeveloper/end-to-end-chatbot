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
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template, footer

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
    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.5)
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
        page_title="Chat with your SQL Server database CSV files",
        page_icon=":Database:"
    )

    #creating a title for the app
    st.title("Chat Application with local Database using the generative ai, using gpt-4-turbo model")
    #footer
    st.markdown(footer,unsafe_allow_html=True)

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
    else:
        st.warning("Please upload a csv file!")
    
    user_question = st.text_input("Ask a question about your documents:")
    with st.spinner("Processing"):
        if user_question and uploaded_files:
            handle_userinput(user_question)
        

if __name__ == '__main__':
    main()