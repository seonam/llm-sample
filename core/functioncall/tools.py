from pydantic import BaseModel, Field

class Naver_News_Search(BaseModel):
    """query를 이용하여 네이버 뉴스를 관련도 순으로 검색합니다."""
    query: str= Field(description="""검색 키워드
규칙:
1. 최대 2개 단어로 구성
2. 불필요한 조사나 형용사 제외
3. 핵심 명사만 포함

예시:
- (좋음) "개봉영화", "영화"
- (나쁨) "새로 개봉한 영화", "요즘 인기있는 영화"
""")

class Tavily_Search(BaseModel):
    """Tavily 검색 엔진을 이용해 일반 텍스트 웹 검색을 수행합니다."""
    query: str = Field(description="""검색 키워드(영어로 변환 선호)""")
    max_results: int = Field(description="""검색 결과 문서의 수,
특별한 요청이 없으면 5, 최소 3에서 최대 20개""")