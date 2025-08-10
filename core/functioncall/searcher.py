from tavily import TavilyClient

import os
import requests

class NaverSearcher:
    def get_news(self, query):
        url = "https://openapi.naver.com/v1/search/news.json"
        headers = {
            "X-Naver-Client-Id": os.getenv('NAVER_CLIENT_ID'),
            "X-Naver-Client-Secret": os.getenv('NAVER_CLIENT_SECRET')
        }
        params = {
            "query": query,
            "display": 30,
            "start": 1,
            "sort": "sim"
        }

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        result = []
        for item in data.get('items', []):
            title = item.get('title', '').replace('&quot;', '"').replace('&lt;', '<').replace('&gt;', '>').replace('<b>', '').replace('</b>', '')
            link = item.get('link', '')
            description = item.get('description', '').replace('&quot;', '"').replace('&lt;', '<').replace('&gt;', '>').replace('<b>', '').replace('</b>', '')
            result.append(f"---\n제목: {title}\nURL: {link}\n내용: {description}\n---")

        return '\n'.join(result)

class TavilySearcher:
    tavily = TavilyClient()

    def tavily_search(self, query, max_results = 5):
        response = self.tavily.search(
        query = query,
        max_results = max_results,
        )
        # results 값을 하나의 문자열로 결합
        return ' \n---\n '.join(["Title: {title} \n URL: {url} \n Content: {content}".format(**i) for i in response['results']])

naverSearcher = NaverSearcher()
tavilySearcher = TavilySearcher()