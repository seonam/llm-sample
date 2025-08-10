import openai

from openai import pydantic_function_tool
from core.functioncall.tools import Naver_News_Search, Tavily_Search

class OpenaiClient:
    client = openai.OpenAI()
    tools = [pydantic_function_tool(Naver_News_Search), pydantic_function_tool(Tavily_Search)]

    def chat(self, msg, model = 'gpt-4.1-mini'):
        messages = [{'role':'user','content':msg}]

        response = self.client.chat.completions.create(
            model = model,
            messages = messages,
            temperature = 0.2,
            max_tokens = 4096
        )
        return response.choices[0].message.content
    
    def chatWithTools(self, messages, stream=False, model = 'gpt-4.1-mini', parallel_tool_calls = True):
        response = self.client.chat.completions.create(
            model = model,
            messages = messages,

            # 사용할 툴 목록 전달
            tools = self.tools,
            # 'auto' : 자율적 툴 판단
            # 'none'이면 툴 사용하지 않음
            # 'required'면 무조건 툴 사용
            tool_choice = 'auto',
            temperature= 0.1,
            max_tokens= 1024,
            stream = stream,
            # 툴 동시 실행 대신 번갈아 실행하기 (Tool-->ToolMsg-->Tool-->ToolMsg-->...)
            parallel_tool_calls=parallel_tool_calls
        )

        if stream: 
            return response
        
        return response.choices[0].message

openaiClient = OpenaiClient()