import re
import html5lib
import bs4
import requests
import time
import pandas as pd
from tqdm import tqdm

# "categoryurl.csv"には5chのカテゴリ一覧(https://menu.5ch.net/bbstable.html)に記載されているリンクが格納されている
df = pd.read_csv('catagoryurl.csv')

# リンク先のページに過去ログのリンクがある場合はそのリンクを"urllist.csv"に，そうでなければ"headline.csv"に出力
url = []
headline = []
for link in tqdm(df['url']):
    res = requests.get(link)
    soup = bs4.BeautifulSoup(res.content, 'html5lib')
    search = re.compile('過去ログ')
    a = soup.find_all(text=search)
    if len(a) != 0:
        link = link + 'kako/kako0000.html'
        url.append(link)
    else:
        headline.append(link)
    time.sleep(1)

df = pd.DataFrame(url,columns=['url'])
df_2 = pd.DataFrame(headline,columns=['url'])
df.to_csv('urllist.csv',index=False)
df_2.to_csv('headlineurl.csv',index=False)
print(headline)