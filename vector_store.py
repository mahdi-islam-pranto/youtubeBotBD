from get_transcript import fetch_transcript
from langchain_text_splitters import RecursiveCharacterTextSplitter

# yt video url
yt_url = "https://www.youtube.com/watch?v=0nhkU_DImhU"
# fetch transcript
transcript = fetch_transcript(yt_url)

# split transcript into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=["\n", " ", "."],
)

# split transcript into chunks
transcript_chunks = text_splitter.split_text(transcript)
print(len(transcript_chunks))
# print(transcript_chunks[0])
