from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from embed import build_retriever
from langchain_core.output_parsers import StrOutputParser

def format_context(docs):
    formatted = []
    for doc in docs:
        ts = doc.metadata.get("timestamp", "unknown")
        formatted.append(f"[{ts}] {doc.page_content}")
    return "\n\n".join(formatted)

def rewrite_question(question: str) -> list[str]:
    rewritten = rewrite_chain.invoke({"question": question})
    queries = [q.strip() for q in rewritten.split("\n") if q.strip()]
    return queries

def retrieve_with_rewrites(question, retriever):
    queries = rewrite_question(question)
    print(queries)

    all_docs = []
    seen_ids = set()

    for q in queries:
        docs = retriever.invoke(q)
        for doc in docs:
            doc_id = doc.metadata.get("id") or doc.page_content[:50]
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append(doc)

    return all_docs

model = OllamaLLM(model="llama3.2")

template = """
You are an assistant that answers questions about YouTube videos based on provided context.
Always include timestamps in your answers.

Context: {context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

rewrite_prompt = ChatPromptTemplate.from_template(
"""
Rewrite the following question into 3 different search queries.
Each query should use different wording but keep the same meaning.
Return each query on a new line. Return only the questions, no additional text.

Question: {question}
""")

rewrite_chain = rewrite_prompt | model | StrOutputParser()

youtube_url = input("Paste YouTube video link: ").strip()
retriever = build_retriever(youtube_url)
print("Transcript embedded. Ask questions.")

while True:
    print("\n\n--------------------------------")
    question = input("Ask your question (q to quit): ")
    if question == "q":
        break

    docs = retrieve_with_rewrites(question, retriever)
    context = format_context(docs)
    print(context)
    result = chain.invoke({"context": context, "question": question})
    print("--------------------------------\n")
    print(result)