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

# 5chの過去ログのurlからコメントのみを抽出
df = pd.read_csv('8url.csv')
a = []
for i, link in enumerate(df['url']):
    try:
        res = s.get(link)
        soup = bs4.BeautifulSoup(res.content, 'html5lib')
        html = soup.find_all('span',attrs={'class' :'escaped'})
        if len(html) >0:
            for tag in html:
                if tag.text != None:        
                    a.append(tag.text)
    except Exception as e:
        print(e)

df_2 = pd.DataFrame(a,columns=['text'])
df_2.to_csv('8url5chcorpus.csv',index=False)