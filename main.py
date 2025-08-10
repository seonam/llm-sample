from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model = 'gpt-4.1-mini', temperature=0.1, max_tokens=1024)


from langchain.prompts import ChatPromptTemplate


from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

recipe_template=ChatPromptTemplate([
    ('system','당신은 전세계의 이색적인 퓨전 조리법의 전문가입니다.'),
    ('user','''저는 {ingredient}를 이용한 환상적인 퓨전 다이닝을 만들고 싶습니다. 추천해주세요!
레시피에 대한 정보를 JSON 형식으로 출력해주세요.''')
])
from langchain.schema.runnable import RunnablePassthrough

prompt1 = ChatPromptTemplate(["{director}의 대표 작품은 무엇입니까?"])
chain1 = (
    prompt1
    | llm
    | StrOutputParser())
    # | {'answer': RunnablePassthrough()})

response = chain1.invoke("스티븐 스필버그")
print(response)
