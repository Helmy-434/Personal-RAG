import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

try:
    from src.loaders import clean_source_name
except ImportError:
    from loaders import clean_source_name

load_dotenv()

MAX_DISTANCE = 0.9

llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_CHAT_MODEL", "google/gemma-4-26b-a4b-it:free"),  
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

PROMPT = """You answer questions about me using ONLY the context below.
- Cite the source for each claim like [cv.pdf, p.2]. Only use the file name, never full file paths.
- If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}"""


def retrieve(store, question, k=4):
    results = store.search_with_scores(question, k=k)
    return [(doc, score) for doc, score in results if score <= MAX_DISTANCE]


def generate(question, results):
    if not results:
        return "I don't have that information in the knowledge base."
    context = "\n\n".join(
        f"[{clean_source_name(d.metadata.get('source', 'unknown'))}, "
        f"p.{d.metadata.get('page', '-')}]\n{d.page_content}"
        for d, _ in results
    )
    return llm.invoke(PROMPT.format(context=context, question=question)).content


def ask(store, question, k=4):
    results = retrieve(store, question, k)
    return generate(question, results), results