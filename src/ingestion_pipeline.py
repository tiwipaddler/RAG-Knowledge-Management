import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

#1. Load documents
def load_documents(docs_path="docs"):
    """Load documents from a directory"""
    print(f"Loading documents from {docs_path}...")

    #Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your pdf files to it.")
    
    #Load documents from directory
    loader = DirectoryLoader(
        path = docs_path, 
        glob="**/*.pdf", 
        loader_cls=PyPDFLoader
    
    )
    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No documents found in {docs_path}. Please add your pdf files to the directory.")

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f" Source {doc.metadata['source']}")
        print(f" Content length: {len(doc.page_content)} characters")
        print(f" Content preview: {doc.page_content[:100]}...")
        print(f" metadata: {doc.metadata}")
       

    return documents

#2. Split documents
def split_documents(documents, chunk_size=800, chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    print(f"Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:

        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f" Source {chunk.metadata['source']}")
            print(f" Content length: {len(chunk.page_content)} characters")
            print(f" Content preview: {chunk.page_content[:50]}...")
            print(f" metadata: {chunk.metadata}")

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks not shown...")

    return chunks

#3. Create embeddings
def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persis ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")

    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")

    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("--- Finished creating vector store ---")

    print(f"Vector store created and stored in {persist_directory}")
    return vectorstore






def main():
    print("Main Function")
    

    #1. Load documents
    documents = load_documents(docs_path="docs")
    print(f"Total pages loaded: {len(documents)}")
    #2. Split documents
    chunks = split_documents(documents)
    print(f"Total chunks: {len(chunks)}")
    #3. Embeddings and Storing in Vector DB
    vectorstore = create_vector_store(chunks)


if __name__ == "__main__":
    main()



