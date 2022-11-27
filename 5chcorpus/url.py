import requests
import bs4
import re
import pandas as pd
import time
import html5lib

from requests.adapters import HTTPAdapter, Retry

# サイトからのボット判定を回避するための設定
s = requests.Session()
retries = Retry(total=5,
                backoff_factor=0.1,
                status_forcelist=[ 500, 502, 503, 504 ])
s.mount('http://', HTTPAdapter(max_retries=retries))

# 5chの各カテゴリの過去ログの一覧のurlから，8つずつのスレッドのリンクを抽出
df = pd.read_csv('urllist.csv')
df = df.dropna()
df = df.reset_index(drop=True)
url = []
for link in df['url']:
    try:
        res = s.get(link)
        soup = bs4.BeautifulSoup(res.content, 'html5lib')
        taglist = soup.find_all('span',attrs={'class' :'title'})
        a = link.split('/')
        a = a[:3]
        a = '/'.join(a)
        if len(taglist) > 0:
            for i, tag in enumerate(taglist):
                atag = tag.find('a')
                if atag !=None:
                    href = atag.get('href')
                    b = a + href
                    url.append(b)
                if i == 7:
                    break
    # 試行中にまれにサイトからボット判定されアクセスエラーが起きたので強制的にエラーを無視させた
    except Exception as e:
        print(e)
    time.sleep(1)
df = pd.DataFrame(url,columns=['url'])
df.to_csv('8url.csv',index=False)