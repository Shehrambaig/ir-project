import httpx
from bs4 import BeautifulSoup
import json
from typing import Dict,Any


class LSEScraper:

    def __init__(self):
        self.base_url="https://api.londonstockexchange.com/api/v1/pages"
        self.headers={
            'accept':'application/json, text/plain, */*',
            'accept-language':'en-US,en;q=0.9',
            'origin':'https://www.londonstockexchange.com',
            'referer':'https://www.londonstockexchange.com/',
            'user-agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        }
    
    async def fetch_form_data(self,news_id: str)->Dict[str,Any]:
        params={
            'path':'news-article',
            'parameters':f'newsId={news_id}'
        }

        async with httpx.AsyncClient() as client:
            response=await client.get(
                self.base_url,
                params=params,
                headers=self.headers,
                timeout=7.0
            )
            response.raise_for_status()
            return response.json()
    
    def extract_text_content(self,api_response: Dict[str,Any])->str:
        html_content=None

        if 'components' in api_response and isinstance(api_response['components'],list):
            for component in api_response['components']:
                if isinstance(component,dict) and 'content' in component:
                    content_list=component['content']
                    if isinstance(content_list,list):
                        for item in content_list:
                            if isinstance(item,dict) and 'value' in item:
                                value_obj=item['value']
                                if isinstance(value_obj,dict):
                                    html_content=value_obj.get('body')
                                    if html_content:
                                        break
                if html_content:
                    break

        soup=BeautifulSoup(html_content,'lxml')

        for script in soup(["script","style"]):
            script.decompose()

        text=soup.get_text()

        lines=(line.strip() for line in text.splitlines())
        chunks=(phrase.strip() for line in lines for phrase in line.split("  "))
        text='\n'.join(chunk for chunk in chunks if chunk)

        return text
    
    async def scrape_and_extract(self, news_id: str) -> str:
        api_response = await self.fetch_form_data(news_id)
        text_content = self.extract_text_content(api_response)
        return text_content

def test_scraper(news_id: str = "17303563"):
    import asyncio
    async def run():
        scraper = LSEScraper()
        result = await scraper.scrape_and_extract(news_id)
        return result
    
    return asyncio.run(run())


if __name__ == "__main__":
    test_scraper()
