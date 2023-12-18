import requests

def scraper(url):
    res=requests.get(url)
    if res.status_code==200:
        print("content")
        print(res.text)
    else:
        print("cannot fetch status code",res.status_code)

urls="https://www.javatpoint.com"
scraper(urls)