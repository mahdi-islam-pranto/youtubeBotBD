from get_transcript import fetch_transcript
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import faiss
from langchain_community.vectorstores import FAISS
load_dotenv()

# yt video url
yt_url = "https://www.youtube.com/watch?v=0nhkU_DImhU"
# fetch transcript
transcript = fetch_transcript(yt_url)

# create text splitter to split transcript into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=["\n", " ", "."],
)

# split transcript into chunks
transcript_chunks = text_splitter.split_text(transcript)
print(f"total chunks created from transcript: {len(transcript_chunks)}")
# print(transcript_chunks[0])

# Convert chunks to Document objects
documents = [Document(
    page_content=chunk, metadata={"source": yt_url}
    ) for chunk in transcript_chunks]


# create embeddings model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",

)

# create vector store
vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)

# save vector store
vector_store.save_local("vector_store")

print("Vector store created")
# print vector store ids
print(vector_store.index_to_docstore_id)
# print vector store first document by id
print(vector_store.docstore.search(vector_store.index_to_docstore_id[0]))

