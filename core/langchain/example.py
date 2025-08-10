from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model = 'gpt-4.1-mini', temperature=0.1, max_tokens=1024)