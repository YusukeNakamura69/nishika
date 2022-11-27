# Nishikaのヘイトスピーチ判定コンペにて公開されているベースラインの手法を利用(https://www.nishika.com/competitions/hate/summary)
# 戦略：事前学習済みモデルをより強いものに変更
# 　　：cross validationを実装
#　 　：訓練データの増強（当ディレクトリに収集データ自体は存在しません）
import os
import random
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
# from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import (
    AutoModel, AutoTokenizer, EvalPrediction, Trainer, 
    TrainingArguments, AutoModelForMaskedLM, AutoModelForSequenceClassification,
)
import datasets
from datasets import Dataset
from pyknp import Juman


# train,testデータの分かち書き
train_df = pd.read_csv("data/train.csv")
# trainデータに5chのクローリングをクラウドソーシングでアノテートしたものを追加

# train_df2 = pd.read_csv('data/cloudtrain.csv')
# train_df = pd.concat([train_df,train_df2])
# train_df = train_df.sample(frac=1,random_state=42,ignore_index=True)

juman = Juman()
seg_list = []
for text in train_df["text"]:
    result = juman.analysis(text)
    seg_list.append(" ".join(mrph.midasi for mrph in result.mrph_list()))
train_df["text"] = seg_list

test_df = pd.read_csv("data/test.csv")
seg_list = []
for text in test_df["text"]:
    result = juman.analysis(text)
    seg_list.append(" ".join(mrph.midasi for mrph in result.mrph_list()))
test_df["text"] = seg_list

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

SEED = 42
seed_everything(SEED)

# 事前学習済みモデルを早大robertaに変更
cfg = {
    # "model_name":"nlp-waseda/roberta-base-japanese",
    "model_name":"nlp-waseda/roberta-large-japanese",
    # 'model_name' : 'cl-tohoku/bert-base-japanese-whole-word-masking',
    "max_length":-1,
    "train_epoch":3,
    "lr":3e-5,
}

tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"])

cfg["max_length"]=128


class HateSpeechDataset(Dataset):
  def __init__(self, X, y=None):
    self.X = X
    self.y = y

  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):
    input = {
        "input_ids": self.X[index]["input_ids"],
        "attention_mask": self.X[index]["attention_mask"],
    }
    
    if self.y is not None:
      input["label"] = self.y[index]

    return input

# trainerの訓練中に評価関数を追跡できるように関数を作成しておきます
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    result = f1_score(p.label_ids, preds)
    return {"f1_score":result}

# batchsizeを増大
trainer_args = TrainingArguments(
    seed=SEED,
    output_dir=".",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_steps=1e6, # saveのステップを大きくしてここではモデルの保存を行わないようにする
    log_level="critical",
    num_train_epochs=cfg["train_epoch"],
    learning_rate=cfg["lr"],
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    save_total_limit=1,
    fp16=True,
    remove_unused_columns=False,
    report_to="none"
)

from sklearn.model_selection import StratifiedKFold

y_preds = []
models = []
oof_train = np.zeros((len(train_df),))

test_X = [tokenizer(text, padding="max_length", max_length=cfg["max_length"], truncation=True) for text in tqdm(test_df["text"])]
test_ds = HateSpeechDataset(test_X)
# cross validationを実装
n=5
kf = StratifiedKFold(n_splits=n, random_state=SEED, shuffle=True)
for train_index, val_index in kf.split(train_df,train_df['label']):
    trn_df = train_df.iloc[train_index]
    val_df = train_df.iloc[val_index]

    trn_X = [tokenizer(text, padding="max_length", max_length=cfg["max_length"], truncation=True) for text in tqdm(trn_df["text"])]
    trn_ds = HateSpeechDataset(trn_X, trn_df["label"].tolist())

    val_X = [tokenizer(text, padding="max_length", max_length=cfg["max_length"], truncation=True) for text in tqdm(val_df["text"])]
    val_ds = HateSpeechDataset(val_X, val_df["label"].tolist())

    trainer = Trainer(
    model=model,
    args=trainer_args,
    tokenizer=tokenizer,
    train_dataset=trn_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    )

    trainer.train()
    test_preds = trainer.predict(test_ds)
    y_preds.append(np.argmax(test_preds.predictions, axis=1))    
    oof_train[val_index] = np.argmax(trainer.predict(val_ds).predictions, axis = 1)

# test.csvの文章に対するヘイトスピーチであるかどうかの予測結果をcsv出力
sub_df = pd.read_csv("data/sample_submission.csv")
y_sub = sum(y_preds) / len(y_preds)
y_sub = (y_sub > 0.5).astype(int)
sub_df["label"] = y_sub
sub_df.to_csv("output/cloud_rob.csv", index=False)
# 確認用にcvスコアを計算し出力
oof_train = (oof_train > 0.5).astype(int)
cv_score = f1_score(train_df['label'].tolist(), oof_train)
print(cv_score)