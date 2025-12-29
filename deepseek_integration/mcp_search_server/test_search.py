import requests
from bs4 import BeautifulSoup
import urllib.parse

def test_baidu_search():
    query = "DeepSeek AI"
    print(f"正在测试百度搜索: {query}...")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    encoded_query = urllib.parse.quote(query)
    url = f"https://www.baidu.com/s?wd={encoded_query}"
    
    try:
        session = requests.Session()
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        results = soup.select(".result.c-container")
        if results:
            print(f"搜索成功！找到 {len(results)} 条结果。前 3 条如下：")
            for i, item in enumerate(results[:3], 1):
                title_tag = item.select_one("h3 a")
                if title_tag:
                    print(f"{i}. {title_tag.get_text().strip()}")
                    print(f"   URL: {title_tag.get('href')}")
        else:
            print("未找到结果，可能是页面结构发生了变化或被拦截。")
    except Exception as e:
        print(f"搜索出错: {e}")

if __name__ == "__main__":
    test_baidu_search()
