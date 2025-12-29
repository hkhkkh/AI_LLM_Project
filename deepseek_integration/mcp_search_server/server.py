from mcp.server.fastmcp import FastMCP
import requests
from bs4 import BeautifulSoup
import json
import urllib.parse
import time

# 创建一个 FastMCP 实例
mcp = FastMCP("WebSearch")

def baidu_search_logic(query: str, max_results: int = 5):
    """
    百度搜索逻辑。
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    encoded_query = urllib.parse.quote(query)
    url = f"https://www.baidu.com/s?wd={encoded_query}"
    
    try:
        # 百度有时需要先访问首页获取 cookie，或者直接带上简单的 headers
        session = requests.Session()
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        results = []
        # 百度搜索结果通常在 div.result.c-container 中
        for item in soup.select(".result.c-container")[:max_results]:
            title_tag = item.select_one("h3 a")
            desc_tag = item.select_one(".c-abstract") or item.select_one(".content_right_abstract")
            
            if title_tag and title_tag.get("href"):
                # 百度链接是加密的，通常需要跳转，这里直接返回加密链接或尝试解析
                results.append({
                    "title": title_tag.get_text().strip(),
                    "href": title_tag.get("href"),
                    "body": desc_tag.get_text().strip() if desc_tag else "无摘要"
                })
        return results
    except Exception as e:
        raise Exception(f"百度搜索失败: {str(e)}")

@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """
    使用百度搜索网页内容。该工具在国内环境非常稳定。
    
    Args:
        query: 搜索关键词
        max_results: 返回的最大结果数量（默认 5）
    """
    print(f"正在通过百度搜索: {query}")
    try:
        results = baidu_search_logic(query, max_results=max_results)
        
        if not results:
            return "未找到相关结果。"
        
        formatted_results = []
        for r in results:
            formatted_results.append(
                f"标题: {r.get('title', '无标题')}\n"
                f"链接: {r.get('href', '无链接')}\n"
                f"摘要: {r.get('body', '无摘要')}\n"
            )
        
        return "\n---\n".join(formatted_results)
    except Exception as e:
        return f"搜索过程中出错: {str(e)}"

if __name__ == "__main__":
    # 运行 MCP 服务器
    mcp.run()

if __name__ == "__main__":
    # 运行 MCP 服务器
    mcp.run()

if __name__ == "__main__":
    # 运行 MCP 服务器
    mcp.run()
