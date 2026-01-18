# Deep Learning Visualization Animations

ディープラーニングの主要なアルゴリズムを視覚的に理解するためのインタラクティブなアニメーション集です。

## ファイル一覧

| ファイル名 | 説明 |
|-----------|------|
| `conv2d_animation.html` | 2D畳み込み（Conv2D）のアニメーション |
| `self_attention_animation.html` | Self-Attentionのアニメーション |
| `multi_head_attention_animation.html` | Multi-Head Self-Attentionのアニメーション |
| `lstm_animation.html` | LSTM（Long Short-Term Memory）のアニメーション |

## 使い方

各HTMLファイルをブラウザで開くだけで動作します。サーバー不要。

### 共通操作

- **Play**: 自動再生
- **Pause**: 一時停止
- **Step**: 1ステップずつ進める
- **Reset**: 最初に戻す
- **Speed**: アニメーション速度調整

---

## 1. Conv2D Animation

2次元畳み込み演算の可視化。

### 設定パラメータ
- Input Size: 5×5 〜 10×10
- Kernel Size: 2×2 〜 4×4
- Stride: 1 または 2

### 表示内容
- 入力行列（Input）
- カーネル（Kernel / Filter）
- 出力行列（Output / Feature Map）
- 計算式のリアルタイム表示

### 数式
```
Output[i][j] = Σ Σ Input[i+m][j+n] × Kernel[m][n]
```

---

## 2. Self-Attention Animation

Transformerの核となるSelf-Attention機構の可視化。

### 設定パラメータ
- Sequence Length: 3〜6
- Embedding Dim: 3〜5

### 処理フェーズ
1. **Phase 1**: Q, K, V の生成 (X·Wq, X·Wk, X·Wv)
2. **Phase 2**: Attention Scores の計算 (Q·K^T / √d_k)
3. **Phase 3**: Softmax の適用
4. **Phase 4**: Output の計算 (Attention · V)

### 表示内容
- 入力行列 (X)
- 重み行列 (Wq, Wk, Wv)
- Query, Key, Value 行列
- Attention Scores
- Attention Weights (softmax後)
- 出力行列
- d_k と √d_k の値

### 数式
```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

---

## 3. Multi-Head Self-Attention Animation

複数のAttention Headを並列に実行するMulti-Head Attentionの可視化。

### 設定パラメータ
- Sequence Length: 3〜5
- d_model: 4, 6, 8
- Num Heads: 2, 4

### 処理フェーズ
1. **Phase 1**: 各ヘッドでQ, K, V生成
2. **Phase 2**: 各ヘッドでAttention Scores計算
3. **Phase 3**: 各ヘッドでSoftmax適用
4. **Phase 4**: 各ヘッドの出力計算 (Attn × V)
5. **Phase 5**: 全ヘッドの出力を連結 (Concat)
6. **Phase 6**: 最終射影 (Concat × Wo)

### 表示内容
- 入力行列 (X)
- 各ヘッドのQ, K, V, Scores, Attention, Output
- 連結結果 (Concat)
- 出力射影行列 (Wo)
- 最終出力
- d_model, num_heads, d_k, √d_k の値

### 数式
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · Wo
where head_i = Attention(X·Wq_i, X·Wk_i, X·Wv_i)
```

---

## 4. LSTM Animation

Long Short-Term Memory (LSTM) のゲート機構と状態更新の可視化。

### 設定パラメータ
- Sequence Length: 3〜6
- Input Dim: 2〜4
- Hidden Dim: 2〜4

### 処理フェーズ（各タイムステップ）
1. **Forget Gate**: 過去の情報をどれだけ忘れるか
2. **Input Gate**: 新しい情報をどれだけ取り込むか
3. **Cell Candidate**: 追加する候補情報
4. **Output Gate**: どの情報を出力するか
5. **Cell State**: セル状態の更新
6. **Hidden State**: 隠れ状態の計算

### 表示内容
- 次元情報 (seq_len, input_dim, hidden_dim, concat_dim)
- 全パラメータ (Wf, Wi, Wc, Wo, bf, bi, bc, bo)
- 入力 (x_t)
- 前の状態 (h_{t-1}, c_{t-1})
- 各ゲートの値
- セル状態・隠れ状態
- シーケンス進行状況

### 数式
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)     ← 忘却ゲート
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)     ← 入力ゲート
c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c) ← セル候補
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t       ← セル状態
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)     ← 出力ゲート
h_t = o_t ⊙ tanh(c_t)                   ← 隠れ状態
```

---

## 技術仕様

- **動作環境**: モダンブラウザ（Chrome, Firefox, Safari, Edge）
- **依存ライブラリ**: なし（Vanilla JavaScript）
- **レスポンシブ**: フレックスボックスによる自動調整

## ライセンス

MIT License
