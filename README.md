# ETSFormer-practice
最先端時系列データ分析モデルETSformerを使ってみる

# 概要
本レポジトリは、Salesforceが2022年に発表した最先端の時系列データ分析モデルである[ETSformer](https://arxiv.org/pdf/2202.01381.pdf)の紹介です。  
多くの時系列タスクでSOTAを達成したと論文には書かれています。まだQiitaやKaggleにはETSformerについての記載が無いです。    
また、Streamlitを用いてウェブアプリ化し、ETSformerを手軽に体験できるようにしました！  
  
# ETSformerとは
ETSformerの論文についてChatGPT先生に聞いてみました。ChatGPT先生は論文の解説も端的でわかりやすいですね！  
※ただし、たまに間違ったことを言うこともあるので注意が必要です。  
***
You:「ETSformer: Exponential Smoothing Transformers for Time-Series Forecasting」という論文について解説してください。  
  
ChatGPT:「ETSformer: Exponential Smoothing Transformers for Time-Series Forecasting」という論文では、時系列データ分析において、Transformers アーキテクチャを採用したモデルを提案しています。  
このモデルは、時系列データを処理するために、Exponential Smoothing (指数平滑) という手法を組み合わせています。指数平滑は、過去のデータを加重平均して、未来のデータを予測する手法です。このモデルでは、指数平滑を行う部分を、Transformers のエンコーダー部分に対応させています。  
また、このモデルでは、指数平滑では表現できないような複雑な時系列パターンも処理できるように、複数の Transformes ブロックを組み合わせることで、より複雑な時系列データを処理することができるようになっています。
このモデルは、様々な時系列データセットを用いて評価されており、他のモデルよりも優れた精度を発揮することが示されています。  
***  

# 手軽にETSformerを体験できるウェブアプリをつくりました！
Streamlitを用いてウェブアプリ化し、ETSformerを体験できるようにしました。  
[こちら](https://isaka-code-etsformer-practice.streamlit.app/)からアプリを体験できます。  
ETSformerは、Exponential Smoothing (指数平滑)とTransformerの合せ技です。パラメータチューニングが少し大変かもしれません。今回は論文のパラメータをそのまま用いています。  

アプリのスクショ  
![アプリのスクショ.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2627015/ebb3806a-f12b-6625-2307-16f3b5945b8a.png)  

# Kaggleで使うには？
実際に使う際は、[ETSformer-pytorch](https://github.com/lucidrains/ETSformer-pytorch)を使用するのが簡単です。  
ただし、Pythonのバージョンが3.8以上である必要があるので、Kaggleのノートブック環境ではPythonのバージョンが足りずインストールできないです。  
GCPを使うか、ローカルで動かす必要があります。  
※2022年12月現在KaggleのPythonのバージョンは3.7.12でダウングレードもアップグレードもできないです。  

# まとめ
このレポジトリでは、最先端の時系列データ分析モデルであるETSformerの紹介を行いました。    
また、それを体感できるウェブアプリを実装しました。   
画像認識や自然言語処理ではディープラーニングがデファクトスタンダードですが、  
時系列タスクにおいては勾配ブースティング系のLightGBMなどがKaggleにおいては猛威を振るっています。  
近い将来に時系列タスクにおいても、ディープラーニングがデファクトスタンダードになるのか楽しみにしています。  


# お知らせ💡
Qiitaに記事を書きました。良かったら見てください！  
[最先端時系列データ分析モデルETSformerを使ってみた](https://qiita.com/Isaka-code/items/848589fc4d7dd153e915)  
  
# Citation
 
- [ETSformer](https://arxiv.org/abs/2202.01381)  
- [ETSformer-pytorch](https://github.com/lucidrains/ETSformer-pytorch)
- [Streamlitで時系列予測アプリを作る](https://rarejob-tech-dept.hatenablog.com/entry/2022/06/10/190000)
```
@misc{woo2022etsformer,
    title   = {ETSformer: Exponential Smoothing Transformers for Time-series Forecasting}, 
    author  = {Gerald Woo and Chenghao Liu and Doyen Sahoo and Akshat Kumar and Steven Hoi},
    year    = {2022},
    eprint  = {2202.01381},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```