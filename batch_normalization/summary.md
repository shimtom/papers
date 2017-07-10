# Batch Normalization
## Paper
* Title: Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
* Authors: Sergey Ioffe, Christian Szegedy
* Link: http://arxiv.org/abs/1502.03167
* Tags: Neural Network, Performance, Covariate Shift, Regularization
* Year: 2015

## Abstract
Deep Neural Netoworkを訓練することは一つ前の層のパラメータが変化すると,各層の入力の分布が訓練中に変化するという事実により複雑化されている.このことは小さな学習係数や注意深いパラメータの初期化を必要とする訓練を遅らせ,非線形性を十分に持つモデルの訓練を難しくすることで悪名高い.
この現象を内部共変量シフトと呼び,この問題に層の入力を正規化することで対処する.この手法ではモデルアーキテクチャの一部を正規化し,また,各訓練用のミニバッチに正規化を行うことで力を発揮する.
Batch Normalization はより大きな学習係数の使用と,初期化に対する注意深さを和らげることを可能とする.
また,正則化としても働き,Dropoutの必要性を排除する場合もある.
最先端の画像分類モデルに適用するとBatch Normalization は14倍も少ない訓練ステップで同じ精度に到達し,また,かなりの差をつけてオリジナルのモデルを打ち負かした.Batch Normalization を適用したモデルの一部を使用して,ImageNet分類での公表されている最良の結果まで精度を向上させた.top-5バリデーションエラーでは4.9%に達し(テストエラーは4.8%),人間の精度を超えた.

## Summary
### Batch Normalization
  ニューラルネットワークor層への入力を正規化することでパフォーマンスの向上を図る.

### アルゴリズム
#### 訓練時
1. 訓練サンプルをベクトル $x_i$ とし,ミニバッチを $B=\{x_1, \cdots, x_m\}$ とする.サンプルの各次元ごとにミニバッチ全体の平均 $\mu_{B}$ と分散 $\sigma^2_B$ を求める.
$$
\begin{aligned}
\mu_B      &= \frac{1}{m}\sum_{i=1}^m x_i \\
\sigma^2_B &= \frac{1}{m}\sum_{i=1}^m \left(x_i - \mu_B \right)^2
\end{aligned}
$$

2. 計算した平均 $\mu_B$ と分散 $\sigma^2_B$ を使用して,入力 $x_i$を正規化する.この時,計算を数値的に安定させるために定数 $\epsilon$ を加える.
$$
\hat{x_i} = \frac{x_i - \mu_B}{\sqrt{\sigma^2_B + \epsilon}}
$$
3. 正規化を行うとその入力が表現しているものが変化してしまう可能性がある.そこで,その表現力を保つために正規化した $\hat{x_i}$ を $\gamma$,$\beta$ を使用してスケーリングとシフトを行なった $y_i$ を $x_i$ の代わりに使用する.この時,$\gamma,\beta$ は誤差逆伝播法を利用して学習する.
$$
y_i = \gamma \hat{x_i} + \beta
$$

#### 推論時
推論時には正規化をミニバッチ(標本集合)ではなく母集合を用いて行う.
$$
\hat{x} = \frac{x - E[x]}{\sqrt{Var[x] - \epsilon}}
$$
$Var[x]$は不偏分散を用いて$Var[x]=\frac{m}{m-1}\cdot E_{B}[\sigma^2_{B}]$として見積もる.
期待値はサイズ$m$の訓練ミニバッチ全体に対するもので$\sigma^2_{B}$はそれらの分散.
移動平均を期待値の代わりに用いることで見積もることができる.
#### 注意
1. 畳み込み層でBatch Normalizationを行う時には,入力画像のサイズを $p\times q$ とした時,ミニバッチの大きさを $m' = m \cdot pq$ とし,各チャネルごとに正規化を行う.
2. Batch Normalizationは非線形変換の直前で使用する.
  $$
  z = g(Wu + b)
  $$
  ここで,$g(\cdot)$は非線形な活性化関数,$W,b$は学習パラメータ,$u$は層の入力.この時,$u$も正規化できるが$u$は非線形変換の出力であるので,その分布の形状は訓練中に変化しやすい.一方で$Wu+b$はよりガウス分布に近いのでこれを正規化することは安定した分布を持った活性値を生み出しやすい.
3. Batch Normalizationを$Wu+b$に適応する際には$b$を無視できる.
  Batch Normalizationで学習するパラメータ$\beta$で吸収できるから.

### 理論
* ニューラルネットワークでの学習は,内部共変量シフト(Internal Covariate Shift)が原因で複雑.
* 内部共変量シフトは,ある層への入力分布が訓練中に変化すること.これが発生する原因は,訓練時に入力となるミニバッチごとに分布が異なっているから.
* 学習時に,内部共変量シフトが起こるとネットワークが異なる入力の分布に適応しようとするために学習が遅くなる.
* 内部共変量シフトを削減し,入力分布を一定にすることができれば変化する分布への適応をへらすことができるため学習を加速させることができる

### 効果
* 学習の加速
* 大きな学習係数を使用することができる
  - Batch Normalization がパラメータの小さな変化を活性値の勾配の準最適化の強調から防ぐから
* 訓練をパラメータのスケールに対してより柔軟にする
  - Batch Normalizationを使用して重みを逆伝播させるときにパラメータのスケールの影響を受けないから
* ヤコビアンを1に近づける
* モデルを正則化する
  - ミニバッチ中の訓練サンプルはミニバッチ中の他のサンプルと関係しており,訓練中のネットワークは入力に対して決まった出力を返さなくなり,これがネットワークの一般化に有効.


## ノート
### (1) Introduction
  - ミニバッチは勾配が訓練セット全体の勾配の近似となること,メモリ量の節約,並列計算による効率性などの利点がある
  - SGDは単純で効果的だが重みの初期値や,特に学習係数の調整が難しい
  - NNの訓練は各層がそれ以前の層のパラメータの変化に大きく影響されるため複雑
  - 学習システムの入力分布が(特に訓練時とテスト時で)変化することを共変量シフト(covariate shift)という.この考えはNN全体や,NNの層にも適用できる
  - 共変量シフトを解消し,入力分布を固定することは良い効果をもたらす
  - 活性化関数にsigmoid関数($f(x)-\frac{1}{1+\exp(-x)}$)を使用すると,$|x|$が大きくなると勾配が$0$に近づき,勾配消失が起こる.一般にはRelU関数($f(x)=\max(0, x)$)で対処されるが,入力分布を固定できれば解消されるのでは
  - NNの内部で分布が変化することを内部共変量シフト(internal covariate shift)と名付ける.
  - Batch Normalization は内部共変量シフトを削減することでNNの訓練を劇的に加速させる手法
  - Batch Normalization では勾配のパラメータのスケールや初期値への依存が減るために学習係数により大きな値の使用やモデルの正則化とDropoutの必要性の削減,飽和するような非線形な関数(e.g. sigmoid)を使用可能にするなどの効果を発揮する.
### (2) Towards Reducing Internal Covariate Shift
  - 訓練をよくするためには,内部共変量シフトの削減が必要
  - NNの入力をホワイトニング(e.g. 平均 0 分散 1となるように線形変換をする)すれば収束性能は向上する.ホワイトニングを各層の入力に対して行うことで内部共変量シフトを削減する.しかし,NNのパラメータをホワイトニングしても,勾配降下法が正規化を香料できていないため,最適化できない
  - loss に関わるNNのパラメータに正規化を説明させる
  - 訓練サンプル全体の共分散などを利用しての正規化は計算コストが高いため,NNの表現力を維持したまま訓練データ全体と関係した訓練サンプルの正規化の方法を考える
### (3) Normalization via Mini-Batch Statistics
  * (3.1) Training and Inference with Batch-Normalized Networks
  * (3.2) Batch-Normalized Convolutional Networks
  * (3.3) Batch Normalization enables higher learning rates
  * (3.4) Batch Normalization regularizes the model
### (4) Experiments
  * (4.1) Activations over time
  * (4.2) ImageNet classification
### (5) Conclusion
