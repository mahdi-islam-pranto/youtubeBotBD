import streamlit as st
from get_transcript import fetch_transcript
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="YouTube Bangla Chatbot", layout="wide")

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Logo_of_YouTube_%282015-2017%29.svg/2560px-Logo_of_YouTube_%282015-2017%29.svg.png", width=60)
    # st.markdown("## YT Bangla Chatbot")
    st.markdown(
        """
        - ইউটিউব ভিডিও লিংক দিন
        - ভিডিও প্রসেস হলে প্রশ্ন করুন
        - সব উত্তর বাংলায় পাবেন!
        """
    )
    st.markdown("---")
    st.markdown('Contact me - [Mahdi Islam Pranto](https://www.linkedin.com/in/mahdi-islam-pranto/)', unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align: center; color: #FF0000;'>📺 YT Bangla Chatbot</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='text-align: center; color: #555;'>যেকোনো ইউটিউব ভিডিও নিয়ে বাংলায় চ্যাট করুন!</p>",
    unsafe_allow_html=True,
)

# User input for YouTube link
yt_url = st.text_input("ইউটিউব ভিডিও লিংক দিন:", placeholder="https://www.youtube.com/watch?v=...")

# Only process transcript and vector store if yt_url changes
if yt_url:
    if (
        "last_url" not in st.session_state
        or st.session_state.last_url != yt_url
        or "vector_store" not in st.session_state
    ):
        with st.spinner("ভিডিও প্রসেস করা হচ্ছে..."):
            transcript = fetch_transcript(yt_url)
            if not transcript:
                st.error("ট্রান্সক্রিপ্ট পাওয়া যায়নি। অনুগ্রহ করে সঠিক ইউটিউব লিংক দিন।")
                st.session_state.transcript_ready = False
            else:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=200,
                    separators=["\n", " ", "."],
                )
                transcript_chunks = text_splitter.split_text(transcript)
                documents = [Document(page_content=chunk, metadata={"source": yt_url}) for chunk in transcript_chunks]
                embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
                vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)
                retriever = vector_store.as_retriever(search_kwargs={"k": 6})
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
                st.session_state.vector_store = vector_store
                st.session_state.retriever = retriever
                st.session_state.prompt = prompt
                st.session_state.llm = llm
                st.session_state.last_url = yt_url
                st.session_state.messages = []
                st.session_state.transcript_ready = True
    else:
        # Already processed for this URL
        pass

    if st.session_state.get("transcript_ready", False):
        st.success("✅ ভিডিও প্রস্তুত! এখন প্রশ্ন করুন।")

        # Chat input area using a form for safe state handling
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("আপনার প্রশ্ন লিখুন:", key="user_input", placeholder="আপনার প্রশ্ন...")
            submitted = st.form_submit_button("✉️")

        if submitted and user_input.strip():
            with st.spinner("উত্তর তৈরি হচ্ছে..."):
                context_docs = st.session_state.retriever.invoke(user_input)
                final_prompt = st.session_state.prompt.invoke({"question": user_input, "context": context_docs})
                response = st.session_state.llm.invoke(final_prompt)
                st.session_state.messages.append(("user", user_input))
                st.session_state.messages.append(("assistant", response.content))

        # Chat history display
        st.markdown("---")
        st.markdown("### 💬 চ্যাট হিস্ট্রি")
        for role, msg in st.session_state.messages:
            if role == "user":
                st.markdown(
                    f"""
                    <div style='background-color:#1976d2; padding:10px; border-radius:10px; margin-bottom:5px;'>
                    <b>🧑‍💻 আপনি:</b> {msg}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(
                    f"""
                    <div style='background-color:#fff8e1; color:#333; padding:10px; border-radius:10px; margin-bottom:10px;'>
                    <b>🤖 চ্যাটবট:</b> {msg}
                    </div>
                    """, unsafe_allow_html=True)
else:
    st.info("ভিডিও লিংক দিন এবং বাংলায় চ্যাট শুরু করুন।")