# embed.py
import re
import shutil
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi

DB_LOCATION = "./chroma_db"

def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/)([^&?/]+)", url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    return match.group(1)

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

    if current_text:
        chunks.append({
            "text": " ".join(current_text),
            "start_seconds": start_time,
            "start_timestamp": seconds_to_timestamp(start_time)
        })

    return chunks

def build_retriever(youtube_url: str):
    # clear previous DB
    shutil.rmtree(DB_LOCATION, ignore_errors=True)

    video_id = extract_video_id(youtube_url)
    yt_api = YouTubeTranscriptApi()
    transcript = yt_api.fetch(video_id).to_raw_data()
    chunks = chunk_transcript(transcript)

    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    documents = []
    ids = []

    for i, chunk in enumerate(chunks):
        documents.append(
            Document(
                page_content=chunk["text"],
                metadata={
                    "timestamp": chunk["start_timestamp"],
                    "video_id": video_id
                },
                id=str(i)
            )
        )
        ids.append(str(i))

    vectordb = Chroma(
        collection_name="youtube_transcripts",
        persist_directory=DB_LOCATION,
        embedding_function=embeddings
    )

    vectordb.add_documents(documents, ids=ids)

    return vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3}
    )