from email import message
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

# Connect to databse of vectors
persistent_directory = "db/chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# LLM of choice
model = ChatAnthropic(model="claude-sonnet-4-5")

# Store conversation as messages
chat_history = []

def ask_question(user_question):
    print(f"\n--- you asked: {user_question} ---")

    # Making sure the question is clear based on previous question history (STEP 1)
    if chat_history:
        # Ask LLM to make the question standalone
        message = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Return only the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New questions: {user_question}")
        ]

        result = model.invoke(message)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    # STEP 2 Need to find the relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    # Print if needed. This is not necesary but good to see
    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        # Show previerw of the first 2 lines of each document
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f" Doc{i}: {preview}...")

    # STEP 3 New final prompt is created
    combined_input = (
        f"Based on the following documents, please answer this question: {user_question}\n\n"
        "Documents:\n"
        + "\n".join([f"- {doc.page_content}" for doc in docs])
        + "\n\nPlease provide a clear, helpful answer using only the information from these documents. "
        "If you can't find the answer in the documents, say "
         '"I do not have enough information to answer that question based on the provided documents."'
    )

    # STEP 4 get the answer
    message = [
    SystemMessage(content="You are a helpful assistant who answers questions based on the provided document and conversation."),
    ] + chat_history + [
    HumanMessage(content=combined_input)
    ]

    result = model.invoke(message)
    answer = result.content

    # STEP 5 Remember the conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")
    return answer




def start_chat():
    print("Ask me questions! Type 'quit' to exit.")

    while True:
        question = input("\nYour question: ")

        if question.lower() == 'quit':
            print("Thank You & Goodbye!")
            break

        ask_question(question)

if __name__ == "__main__":
    start_chat()



