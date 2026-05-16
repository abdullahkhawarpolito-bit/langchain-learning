from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
)

model = ChatHuggingFace(llm=llm)
chat_history=[
    SystemMessage(content="You are a helpful assistant.")
]

while True:
    user_input = input("You: ")
    chat_history.append( HumanMessage(content=user_input))


    if user_input == "exit":
        print("Exiting the chatbot. Goodbye!")
        break

    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))

    print(f"Chatbot: {result.content}")

print(chat_history)