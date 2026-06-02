from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
    
persistent_directory = "db/chroma_db"

#Load embeddings and vector store
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings_model,
    collection_metadata={"hnsw:space": "cosine"}
)

#Search for relevant documents
query = "What types of taxes are taught and how should they be taught?"

#retriever = db.as_retriever(search_kwargs={"k": 5})

#setting the quality of the retriever. Make sure to comment out the above retriever if using this one. Use Cmd + /
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.3   # Only returns chunks with cosine similarity => 0.3
    }
)

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
#Here we are displaying the results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")


# Combine the query and the relevant document contents
combined_input = f"""Based on the following documents, please answer this questions: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I do not have enough information to answer that question based on the provided documents."
"""

# Chat with the chosen LLM
model = ChatAnthropic(model="claude-sonnet-4-5")

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
result = model.invoke(messages)

#Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)

