from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from embed import retriever

def format_context(docs):
    formatted = []
    for doc in docs:
        ts = doc.metadata.get("timestamp", "unknown")
        formatted.append(f"[{ts}] {doc.page_content}")
    return "\n\n".join(formatted)

model = OllamaLLM(model="llama3.2")

template = """
You are an assistant that answers questions about YouTube videos based on provided context.
Always include timestamps in your answers.

Context: {context}

Question: {question}

Answer format:
- Short answer
- Bullet points
- Timestamps
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n--------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n")
    if question == "q":
        break

    docs = retriever.invoke(question)
    context = format_context(docs)
    result = chain.invoke({"context": context, "question": question})
    print(result)