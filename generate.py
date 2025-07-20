from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()


# create embeddings model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",

)

# load vector store
DB_FAISS_PATH="vector_store"
db = FAISS.load_local(DB_FAISS_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

# similar_docs = db.similarity_search(
#     "what naval say about wealth creation?",
#     k=6,
#     )


# make prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "তুমি একজন সহায়ক ইউটিউব চ্যাটবট, যেটি ইউটিউব ভিডিওর ট্রান্সক্রিপ্ট ব্যবহার করে প্রশ্নের উত্তর দাও। সব উত্তর বাংলায় দেবে।"),
        ("user", "প্রশ্ন: {question}"),
        ("user", "প্রাসঙ্গিক ইউটিউব ভিডিওর ট্রান্সক্রিপ্ট: {context}"),
        ("assistant", 
         "উপরের ট্রান্সক্রিপ্ট এবং প্রশ্নের ভিত্তিতে বাংলায় বিস্তারিত ও সহজভাবে উত্তর দাও। "
         "যদি ট্রান্সক্রিপ্টের সাথে প্রশ্নের মিল না পাও, তাহলে ইউটিউব ভিডিওর বিষয়বস্তুর সাথে সম্পর্কিত একটি প্রাসঙ্গিক প্রশ্ন বাংলায় জিজ্ঞাসা করো, "
         "যাতে ব্যবহারকারী আরও স্পষ্টভাবে জানাতে পারে সে কী জানতে চায়।")
    ]
)


# make a retriever 
retriever = db.as_retriever(search_kwargs={"k": 6})

user_question = "what naval say about wealth creation?"
full_context = retriever.invoke(user_question)

final_prompt = prompt.invoke({"question": user_question, "context": full_context})

response = llm.invoke(final_prompt)

# generate response as text file
with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response.content)


print(response.content)