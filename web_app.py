import streamlit as st
from get_transcript import fetch_transcript
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()
import os

st.set_page_config(page_title="YouTube Bangla Chatbot", layout="wide")
st.title("📺 ইউটিউব বাংলা চ্যাটবট")

# User input for YouTube link
yt_url = st.text_input("ইউটিউব ভিডিও লিংক দিন:")

if yt_url:
    with st.spinner("ট্রান্সক্রিপ্ট প্রসেস করা হচ্ছে..."):
        transcript = fetch_transcript(yt_url)
        if not transcript:
            st.error("ট্রান্সক্রিপ্ট পাওয়া যায়নি।")
            st.stop()

        # Split transcript
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n", " ", "."],
        )
        transcript_chunks = text_splitter.split_text(transcript)
        documents = [Document(page_content=chunk, metadata={"source": yt_url}) for chunk in transcript_chunks]

        # Embeddings and vector store
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 6})

        # Prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "তুমি একজন সহায়ক ইউটিউব চ্যাটবট, যেটি ইউটিউব ভিডিওর ট্রান্সক্রিপ্ট ব্যবহার করে প্রশ্নের উত্তর দাও। সব উত্তর বাংলায় দেবে।"),
                ("user", "প্রশ্ন: {question}"),
                ("user", "প্রাসঙ্গিক ট্রান্সক্রিপ্ট: {context}"),
                ("assistant", 
                 "উপরের ট্রান্সক্রিপ্ট এবং প্রশ্নের ভিত্তিতে বাংলায় বিস্তারিত ও সহজভাবে উত্তর দাও। "
                 "যদি ট্রান্সক্রিপ্টের সাথে প্রশ্নের মিল না পাও, তাহলে ইউটিউব ভিডিওর বিষয়বস্তুর সাথে সম্পর্কিত একটি প্রাসঙ্গিক প্রশ্ন বাংলায় জিজ্ঞাসা করো, "
                 "যাতে ব্যবহারকারী আরও স্পষ্টভাবে জানাতে পারে সে কী জানতে চায়।")
            ]
        )
        llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

    # Chat interface
    st.success("ভিডিও প্রস্তুত! এখন প্রশ্ন করুন।")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.text_input("আপনার প্রশ্ন লিখুন:", key="user_input")
    if st.button("জিজ্ঞাসা করুন") and user_input:
        with st.spinner("উত্তর তৈরি হচ্ছে..."):
            context_docs = retriever.invoke(user_input)
            final_prompt = prompt.invoke({"question": user_input, "context": context_docs})
            response = llm.invoke(final_prompt)
            st.session_state.messages.append(("user", user_input))
            st.session_state.messages.append(("assistant", response.content))

    # Display chat history
    for role, msg in st.session_state.messages:
        if role == "user":
            st.markdown(f"**আপনি:** {msg}")
        else:
            st.markdown(f"**চ্যাটবট:** {msg}")