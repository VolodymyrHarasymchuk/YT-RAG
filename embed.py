from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi

def seconds_to_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}"

def chunk_transcript(transcript, max_words=200):
    chunks = []
    current_text = []

    start_time = transcript[0]["start"]

    for seg in transcript:
        current_text.append(seg["text"])

        if len(" ".join(current_text).split()) >= max_words:
            chunks.append({
                "text": " ".join(current_text),
                "start_seconds": start_time,
                "start_timestamp": seconds_to_timestamp(start_time)
            })

            current_text = []
            start_time = seg["start"]

    # handle leftover text
    if current_text:
        chunks.append({
            "text": " ".join(current_text),
            "start_seconds": start_time,
            "start_timestamp": seconds_to_timestamp(start_time)
        })

    return chunks

ytt_api = YouTubeTranscriptApi()
transcript = ytt_api.fetch("LPZh9BOjkQs").to_raw_data()

chunks = chunk_transcript(transcript)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db_location = "./chroma_db"

documents = []
ids = []

for i, chunk in enumerate(chunks):
    document = Document(
        page_content=chunk["text"],
        metadata={"timestamp": chunk["start_timestamp"]},
        id=str(i)
    )
    ids.append(str(i))
    documents.append(document)

vectordb = Chroma(
    collection_name="youtube_transcripts",
    persist_directory=db_location,
    embedding_function=embeddings
)

vectordb.add_documents(documents, ids=ids)
retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)