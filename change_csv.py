import pandas as pd
import os

df = pd.read_csv("train.csv")
df['filename'] = df['filename'].apply(lambda x: os.path.join(os.getcwd(),'characters/') + x.rsplit("/",2)[1] + '/' + x.rsplit("/",1)[1])
df.to_csv("train.csv",index=False)

df = pd.read_csv("eval.csv")
df['filename'] = df['filename'].apply(lambda x: os.path.join(os.getcwd(),'eval_images/') + x.rsplit("/",2)[1] + '/' + x.rsplit("/",1)[1])
df.to_csv("eval.csv",index=False)
