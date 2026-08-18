import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from langchain_openai import ChatOpenAI

load_dotenv()

# ------------------------
# Load PDF
# ------------------------

loader = PyPDFLoader("pdf\Pratikesh_Howale.pdf")
documents = loader.load()

# ------------------------
# Split
# ------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=50,
)

chunks = splitter.split_documents(documents)

print(f"Chunks: {len(chunks)}")

# ------------------------
# Embeddings
# ------------------------

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ------------------------
# FAISS
# ------------------------

db = FAISS.from_documents(chunks, embedding)

retriever = db.as_retriever(search_kwargs={"k":3})

# ------------------------
# LLM
# ------------------------

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

SYSTEM_PROMPT = """
You are an AI assistant.

Use ONLY the supplied context.

If the answer isn't present, reply:

I don't know.

Context:

{context}
"""

print("\nLocal PDF Chatbot")
print("---------------------------")

while True:

    question = input("\nYou : ")

    if question.lower()=="exit":
        break

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = f"""
{SYSTEM_PROMPT}

Question:

{question}

Answer:
"""

    response = llm.invoke(prompt)

    print("\nAssistant:\n")

    print(response.content)

    print("\nSources:")

    for d in docs:
        print(
            f"Page {d.metadata['page']}"
        )