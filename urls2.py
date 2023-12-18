import requests
from bs4 import BeautifulSoup

def scraper(url):
    res=requests.get(url)
    if res.status_code==200:
        print("contents")
        soup=BeautifulSoup(res.content,'html.parser')
        print(soup.get_text())
    else:
        print("not",res.status_code)
urls="https://www.javatpoint.com"
scraper(urls)
