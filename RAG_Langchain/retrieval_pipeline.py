from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# Configuration
PERSISTENT_DIRECTORY = "db/chroma_db"


#====================================================================================================================================


# Step-1: Add embedding and store vector
# Load embedding and vector store
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=PERSISTENT_DIRECTORY,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

# Query
query = "tell me about post covid conditions"



#====================================================================================================================================


# Step-2: Add Retriever
# Retriever
retriever = db.as_retriever(search_kwargs={"k": 2})

# Retrieved documents from vector database
retrieved_docs = retriever.invoke(query)

print(f"User Query: {query}")
print("--- Context ---")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

# Combine the query and the relevant document contents
prompt = f"""Based on the following context, please answer this question: {query}

contect:
{chr(10).join([f"- {doc.page_content}" for doc in retrieved_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""


#====================================================================================================================================



# Step-3: Add LLM
# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=prompt),
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)
