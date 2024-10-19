import re
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader, AsyncHtmlLoader

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
import time

# 크롤링을 할 때 사용하는 크롬 브라우저의 설정을 정의합니다.
chrome_options = Options()
chrome_options.add_argument("--headless")  # 브라우저를 헤드리스 모드로 실행합니다.
chrome_options.add_argument("--disable-gpu")  # GPU를 사용하지 않도록 설정합니다.
chrome_options.add_argument(
    "--no-sandbox"
)  # 샌드박스 모드를 사용하지 않도록 설정합니다.


class WebsiteDataCrawler:
    """Naver, NCSoft, Anthropic 블로그 사이트에서 블로그 콘텐츠를 크롤링해오는 클래스입니다

    Attributes:
        driver (selenium.webdriver): Javascript가 필요한 웹사이트를 크롤링할 때 필요한 webdriver입니다.
    """

    def __init__(self):
        self.driver = webdriver.Firefox(options=chrome_options)
        pass

    def get_all_hrefs(self, url: str) -> list:
        """
        주어진 URL에서 <a>태그에 있는 모든 href를 크롤링하여 반환합니다.

        Args:
            url (str): href들을 얻고자 하는 웹페이지의 URL.

        Returns:
            list: 웹페이지에 존재하는 모든 href들을 포함하는 리스트.
        """
        # URL에 대한 GET 요청 보내기
        response = requests.get(url)

        # BeautifulSoup을 사용하여 HTML 콘텐츠 파싱
        soup = BeautifulSoup(response.content, "html.parser")

        # 모든 <a> 태그 찾기
        a_tags = soup.find_all("a")

        # 각 <a> 태그에서 href 속성 추출
        hrefs = [a.get("href") for a in a_tags if a.get("href") is not None]

        return hrefs

    def get_anthropic_suburls(self) -> list:
        """
        Anthropic 뉴스 페이지에서 모든 게시물의 URL을 가져옵니다.

        Returns:
            list: 모든 뉴스 게시물의 URL을 포함한 리스트를 반환합니다.
        """
        url = "https://www.anthropic.com/news"
        site_href_list = self.get_all_hrefs(url)
        news_hrefs = [
            url + href.replace("/news", "")
            for href in site_href_list
            if "/news" in href and href != "/news"
        ]
        return list(set(news_hrefs))

    def get_ncsoft_suburls(self) -> list:
        """
        NCSoft 블로그 페이지에서 모든 게시물의 URL을 가져옵니다.

        Returns:
            list: 모든 블로그 게시물의 URL을 포함한 리스트를 반환합니다.
        """
        url = "https://ncsoft.github.io/ncresearch/blogs/"
        site_href_list = self.get_all_hrefs(url)
        pattern = re.compile(r"^/ncresearch/[a-fA-F0-9]{40}$")
        filtered_paths = [
            url.replace("/ncresearch/blogs/", "") + path
            for path in site_href_list
            if pattern.match(path)
        ]
        return filtered_paths

    def get_naver_suburls(self) -> list:
        """
        Naver Tech Blog에서 모든 게시물의 URL을 가져옵니다.

        Returns:
            list: 모든 게시물의 URL을 포함한 리스트를 반환합니다.
        """
        url = "https://clova.ai/tech-blog"
        site_href_list = list(set(self.get_all_hrefs(url)))
        filtered_path = [
            url.replace("/tech-blog", "") + path
            for path in site_href_list
            if ("/tech-blog" in path)
            and ("/tag/" not in path)
            and (path != "/tech-blog")
        ]
        return filtered_path

    def get_all_docs(self) -> list:
        """
        지정된 회사들 (OpenAI, Anthropic, NCSoft, Naver) 웹사이트에서 모든 문서를 가져옵니다.

        Returns:
            list: 모든 문서를 담고 있는 리스트를 반환합니다.
        """
        # 모든 문서 서브 URL을 통해 문서 목록을 가져옵니다
        # openai = self.get_openai_suburls()
        anthropic = self.get_anthropic_suburls()
        ncsoft = self.get_ncsoft_suburls()
        naver = self.get_naver_suburls()

        total_list_suburl = anthropic + ncsoft + naver

        loader = WebBaseLoader(total_list_suburl)
        docs = loader.load()
        return docs


'''
    def get_openai_suburls(self, n_recent: int = 30) -> list:
        """
        OpenAI 웹사이트에서 최근 게시된 뉴스 페이지들의 URL을 반환합니다.

        Args:
            n_recent (int): 최근에 게시된 게시물 중에서 가져올 개수입니다.

        Returns:
            list: 최근에 추가된 뉴스 페이지들의 URL을 포함한 리스트입니다.
        """
        url = f"https://openai.com/news/?limit={n_recent}"
        openai_url_list = list()
        try:
            self.driver.get(url)


            # Javascript가 로드될 수 있도록 기다리는 시간을 설정합니다
            time.sleep(2)

            # Find all <a> tags
            links = self.driver.find_elements(By.TAG_NAME, "a")
            print(links)
            for link in links:
                link_dir = link.get_attribute("href")
                if "/index/" in link_dir:
                    openai_url_list.append(link_dir)
        finally:
            # 크롤링 완료 후 driver 종료합니다
            self.driver.quit()

        return list(set(openai_url_list))
'''

if __name__ == "__main__":
    website_data = WebsiteDataCrawler()
    site_list = website_data.get_all_docs()

    for site in site_list:
        print(site)
