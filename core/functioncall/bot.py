from core.functioncall.searcher import naverSearcher, tavilySearcher
from core.functioncall.openai import openaiClient
import json

class SearchBot:
    available_functions = {'Naver_News_Search': naverSearcher.get_news, 'Tavily_Search': tavilySearcher.tavily_search}

    def __init__(self, openai):
        self.openai = openai

    def search(self, prompt):
        print('Prompt:',prompt)

        tool_call_result = self.openai.chatWithTools([
            {
                "role": "user",
                "content": prompt
            }
        ])
        # 프롬프트를 받아 툴 사용 여부를 결정합니다.
        # 단순 대화라면 tool call을 생성하지 않습니다.

        print('---')
        print('News_Bot: Call ', end='')

        if tool_call_result.tool_calls:
            name, arguments = tool_call_result.tool_calls[0].function.name, tool_call_result.tool_calls[0].function.arguments

            print(name, arguments)

            # 툴 실행
            search_result = self.available_functions[name](**json.loads(arguments))
            print('---')
            print('News_Bot:',end='')

            # 프롬프트 + 툴 요청 + 툴 실행 결과 전달
            response = self.openai.chatWithTools([
                    {
                        "role": "user",
                        "content": prompt
                    },
                    # tool 요청 시, 이전 메세지는 반드시 tool_call_result 이어야 함.
                    tool_call_result,
                    {
                        "role": "tool",
                        "content": search_result,
                        "tool_call_id":tool_call_result.tool_calls[0].id
                    }
                ],
                stream=True
            )

            for chunk in response:
                print(chunk.choices[0].delta.content, end='')
        else:
            print('Nothing')
            print(tool_call_result.content)

    # 하나의 메세지에서 여러 개의 Tool Call 을 수행해야 하는 경우
    # 아래와 같이 한번에 LLM 에게 요청을 하면 입력되는 툴 출력이 너무 길어 할루시네이션이 발생할 수 있다.
    def searchV2(self, prompt):
        print('Prompt:',prompt)
        tool_call_result = self.openai.chatWithTools([
            {
                "role": "user",
                "content": prompt
            },
        ])

        print('---')
        print('News_Bot: Call ', end='')

        if tool_call_result.tool_calls:
            tool_messages=[]

            # 여러 개의 tool_call에 대해, search_result 계산하여 리스트로 저장
            for tool_call in tool_call_result.tool_calls:
                name, arguments = tool_call.function.name, tool_call.function.arguments
                print(name,arguments)

                search_result = self.available_functions[name](**json.loads(arguments))

                print('---')
                print('Search_Bot:',end='')

                tool_messages.append(
                    {"role": "tool","content": search_result,"tool_call_id":tool_call.id}
                )

            print("Call LLM")
            response = self.openai.chatWithTools(
                [   
                    {
                        "role": "user",
                        "content": prompt
                    },
                    tool_call_result
                ] + tool_messages,
                stream=True,
                model = 'gpt-4.1-mini'
            )

            for chunk in response:
                print(chunk.choices[0].delta.content, end='')
        else:
            print('Nothing')
            print(tool_call_result.content)
    
    # parallel_tool_calls 를 비활성화하고, Tool 호출 -> 메세지 전달 -> Tool 호출 -> 형태로 처리한다.
    def searchV3(self, prompt):
        print('Prompt:',prompt)
        msgs = [
            {
                "role": "user",
                "content": prompt
            },
        ]

        tool_call_result = self.openai.chatWithTools(msgs, parallel_tool_calls=False)

        print('---')
        print('Search_Bot: Call ', end='')

        while tool_call_result.tool_calls:
            msgs.append(tool_call_result)
            name, arguments = tool_call_result.tool_calls[0].function.name, tool_call_result.tool_calls[0].function.arguments
            print(name,arguments)

            search_result = self.available_functions[name](**json.loads(arguments))
            print('---')
            print('Search_Bot:',end='')

            msgs.append(
                {"role": "tool","content": search_result,"tool_call_id":tool_call_result.tool_calls[0].id}
            )
            tool_call_result = self.openai.chatWithTools(msgs, parallel_tool_calls=False)
            print("Call LLM")


        print(tool_call_result.content)

searchBot = SearchBot(openaiClient)