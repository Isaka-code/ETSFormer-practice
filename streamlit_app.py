
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch
from etsformer_pytorch import ETSFormer

def setup_ETSformer():
    model = ETSFormer(
        time_features = 1,
        model_dim = 512,                # in paper they use 512
        embed_kernel_size = 3,          # kernel size for 1d conv for input embedding
        layers = 2,                     # number of encoder and corresponding decoder layers
        heads = 8,                      # number of exponential smoothing attention heads
        K = 1,                          # num frequencies with highest amplitude to keep (attend to)
        dropout = 0.2                   # dropout (in paper they did 0.2)
    )
    return model
# ETSformerモデルを読み込み
model = setup_ETSformer()

# streamlitにメッセージ表示
st.title('ETSformerで時系列予測')
st.write("AirPassengers.csv（12年×12ヶ月）を使用します。11年を学習に使い、残り1年を予測します。")
st.write("AirPassengers.csvは、下記の外部サイトよりダウンロードが可能です。")
st.write("https://www.analyticsvidhya.com/wp-content/uploads/2016/02/AirPassengers.csv")

# 時系列データ読み込み
df = pd.read_csv("./data/AirPassengers.csv", delimiter=",")
# 144 = 12年×12ヶ月。11年を学習に使い、残り1年を予測します。 
train_df, val_df = train_test_split(df, train_size=132/144, shuffle=False)
train, val = torch.tensor(train_df["#Passengers"].values, dtype=torch.float), torch.tensor(val_df["#Passengers"].values, dtype=torch.float)
train, val = torch.reshape(train, (1, -1, 1)), torch.reshape(val, (1, -1, 1))

num_steps_forecast = len(val_df)
pred = model(train, num_steps_forecast = num_steps_forecast) # (1, 32, 4) - (batch, num steps forecast, num time features)
pred = torch.reshape(pred, (-1,)).detach().numpy()

# 図の描画
fig, ax = plt.subplots()

ax.plot(list(range(0,len(train_df))), train_df["#Passengers"].values, label="past")
ax.plot(list(range(len(train_df),len(df))), pred, label="prediction") #予測値リストのサイズに合うようにrangeの区間を設定。
plt.grid()

plt.legend()
st.pyplot(fig)