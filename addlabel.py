import pandas as pd
# 5chからクローリングした文章をもとにYahooクラウドソーシングでヘイトスピーチ判定タスクを実施して得た"3589042334.tsv"から文章(=text)と判定結果(=label)を抽出
# skiprowsで読み飛ばした行の内容はタスク回答者の質を保つために用いたチェック設問に対する回答
df = pd.read_csv('3589042334.tsv',sep='\t',header=None,usecols=[3,6],skiprows=2280)
df = df.set_axis(['text','label'],axis=1)

# 1つの文章につき3人がアノテートしている
# labelにはヘイトスピーチであるか，また文章として破綻しているかを問う5つの選択肢から選んだ回答が格納されているため，これを数値に置き換えscoreとする
score = []
for i, label in enumerate(df['label']):
    if label == 'ヘイトスピーチである':
        s=3
        score.append(s)
    if label == 'ややヘイトスピーチである':
        s=2
        score.append(s)
    if label == 'あまりヘイトスピーチでない':
        s=1
        score.append(s)
    if label == 'ヘイトスピーチではない':
        s=0
        score.append(s)
    if label == '文章として破綻，もしくはシステムの説明文':
        s=31.5
        score.append(s)
df['score'] = score

# 各文章につき3人いるアノテーターの回答の平均をとる
ave = []
comment = []
for i, score in enumerate(df['score']):
    if i%3==0:
        comment.append(df['text'][i])
        a = (df['score'][i]+df['score'][i+1]+df['score'][i+2])/3
        ave.append(a)

# 文章として破綻を選んだ人が1人でもいた場合にその文章を省き，残りの文章で平均scoreが2.0を超えたものをラベル1，それ以外をラベル0とする
label = []
for score in ave:
    if score > 10:
        la = ''
    elif score > 2.0:
        la = 1
    else:
        la = 0
    label.append(la)

# nishikaのヘイトスピーチ判定コンペ(https://www.nishika.com/competitions/hate/summary)のtrainデータの形式に合わせてcsv出力
data = list(zip(comment,label))
df_2 = pd.DataFrame(data,columns=['text','label'])
# print(df_2)
df_2.dropna(inplace=True)
df_2['label'] = df_2['label'].astype(int)
df_2.to_csv('cloudtrain.csv',index=False)