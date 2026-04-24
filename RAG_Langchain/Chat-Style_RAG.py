from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables
load_dotenv()

#connect to your Database
persistent_dir = "db/chroma_db"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory=persistent_dir, embedding_function=embeddings)

# Set up AI model
model = ChatOpenAI(model="gpt-4o")

# Store our conversation as messages
chat_history = []

def ask_q(user_q):
    print(f"\n--- You asked -- {user_q}")


    # step1 
    if chat_history:
        #asking llm itself me question stanalone
        msg = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_q}")
        ]
        
        result = model.invoke(msg)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")

    else:
        search_question = user_q

    
    #step 2 : find relevant docs
    retriever = db.as_retriever(search_kwargs={"k":3})
    docs = retriever.invoke(search_question)

    # Step 3: Create final prompt
    prompt = f"""Based on the following context, please answer this question: {user_q}

    Context:
    {"\n".join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """
    
    # Step 4: Get the answer
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=prompt)
    ]
    
    result = model.invoke(messages)
    answer = result.content   



    #step 5: Remember this conversation
    chat_history.append(HumanMessage(content=user_q))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")
    return answer





#chat loop
def start_chat():
    print("ask me questions! type 'quit' to exit")

    while True:
        question = input("\n Your question=>")

        if question.lower() == 'quit':
            print("Goodbye")
            break
        ask_q(question)



if __name__ == "__main__":
    start_chat()