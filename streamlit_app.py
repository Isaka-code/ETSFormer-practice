import streamlit as st

from darts import TimeSeries
from darts.models import ExponentialSmoothing, ARIMA, AutoARIMA, BATS, Theta

from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

st.title('Dartsで時系列予測')

# Dartsの処理
# モデルの選択
model_select = st.radio(
    "使用するモデルを選んでください",
    ("ExponentialSmoothing", "ARIMA", "AutoARIMA", "BATS", "Theta"))
if model_select == "ExponentialSmoothing":
    model = ExponentialSmoothing()
    st.write("確率論的モデルです。サンプリング回数はどうしますか？")
    num_samples = st.number_input("num_samples", value=1)

elif model_select == "ARIMA":
    model = ARIMA()
    st.write("確率論的モデルです。サンプリング回数はどうしますか？")
    num_samples = st.number_input("num_samples", value=1)

elif model_select == "BATS":
    model = BATS()
    st.write("確率論的モデルです。サンプリング回数はどうしますか？")
    num_samples = st.number_input("num_samples", value=1)
# num_samplesを持たないモデル
elif model_select == "AutoARIMA":
    model = AutoARIMA()
    num_samples = 1
elif model_select == "Theta":
    model = Theta()
    num_samples = 1


# csvをアップロード
csv = st.file_uploader('Upload a CSV')

if csv is None:
    st.write("ファイルをアップロードしてください")
else:
    df = pd.read_csv(csv, delimiter=",")
    train, val = train_test_split(df, shuffle=False)
    train, val = TimeSeries.from_dataframe(train, 'Month', '#Passengers'), TimeSeries.from_dataframe(val, 'Month', '#Passengers')

    model.fit(train)
    prediction = model.predict(len(val), num_samples=num_samples)

    low_quantile = st.number_input("low_quantile", value=0.05, format="%f")
    high_quantile = st.number_input("high_quantile", value=0.95, format="%f")

    # 図の描画
    fig, axes = plt.subplots()
    series = TimeSeries.from_dataframe(df, 'Month', '#Passengers')
    series.plot()
    prediction.plot(label='forecast', low_quantile=low_quantile, high_quantile=high_quantile)
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)