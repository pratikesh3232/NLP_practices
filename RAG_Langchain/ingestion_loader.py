import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()







#==========================================================================





# Loading Documents
def load_doc(docs_path = "data"):
    print(f"loading docs from  {docs_path} .. ")


    #check if path exits
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"path not found in {docs_path}")
    

    #Load all docs 
    loader = DirectoryLoader(path =docs_path, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    
    #to check if file found or not
    if len(docs) == 0 :
        raise FileNotFoundError(f"no file foind in {docs_path}")



    # #Just to preview docs 
    # for i, doc in enumerate(docs[:2]):
    #     print(f"\nDocument {i+1}:")
    #     print(f"  Source: {doc.metadata['source']}")
    #     print(f"  Content length: {len(doc.page_content)} characters")
    #     print(f"  Content preview: {doc.page_content[:100]}...")
    #     print(f"  metadata: {doc.metadata}")


    
    return docs







#==========================================================================






#Spliting Or Chunking Documents
def split_docs(docs,chunk_size =100 ,chunk_overlap=10):

    text_splitter= CharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(docs)

    
    # #just to show how chunks look like
    # if chunks:
    
    #     for i, chunk in enumerate(chunks[:5]):
    #         print(f"\n--- Chunk {i+1} ---")
    #         print(f"Source: {chunk.metadata['source']}")
    #         print(f"Length: {len(chunk.page_content)} characters")
    #         print(f"Content:")
    #         print(chunk.page_content)
    #         print("-" * 50)
        
    #     if len(chunks) > 5:
    #         print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks





#==========================================================================


def vectorstore(chunks,persist_directory="db/chroma_db"):
    embd_model = OpenAIEmbeddings(model = "text-embedding-3-small")


    print("---- creating vector Store ----")
    vector_store = Chroma.from_documents(documents=chunks,
                                         embedding=embd_model,
                                         persist_directory=persist_directory,
                                         collection_metadata={"hnsw:space": "cosine"}
                                         )

    return vector_store





























def main():
    docs_path = "data"
    # Step 1: Load documents
    documents = load_doc(docs_path)  

    # Step 2: Split into chunks
    chunks = split_docs(documents)
    
    # # Step 3: Create vector store
    vector_store = vectorstore(chunks)


if __name__ == "__main__":
    main()
