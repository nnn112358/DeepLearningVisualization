# ONNX演算子 - RNN演算 (Recurrent Neural Network Operations)

RNN演算は時系列データやシーケンスデータを処理するための再帰的なニューラルネットワーク層を提供します。自然言語処理や音声認識で広く使用されています。

## 概要図

```mermaid
graph TB
    subgraph "RNN演算の分類"
        A[RNN演算] --> B[LSTM<br/>長短期記憶]
        A --> C[GRU<br/>ゲート付き再帰ユニット]
        A --> D[RNN<br/>シンプルRNN]
    end

    subgraph "特徴比較"
        B --> B1["ゲート: 3<br/>セル状態: あり<br/>パラメータ: 多"]
        C --> C1["ゲート: 2<br/>セル状態: なし<br/>パラメータ: 中"]
        D --> D1["ゲート: 0<br/>セル状態: なし<br/>パラメータ: 少"]
    end
```

---

## LSTM（Long Short-Term Memory）

### 説明
長短期記憶ネットワーク。ゲート機構により長期依存関係を学習できるRNNユニットです。1997年にHochreiterとSchmidhuberが提案し、勾配消失問題を軽減することで長いシーケンスを効果的に処理できます。

### 構造

```mermaid
graph TD
    subgraph "LSTMセルの構造"
        X["入力 x_t"] --> FG[忘却ゲート<br/>f_t]
        X --> IG[入力ゲート<br/>i_t]
        X --> CG[候補セル<br/>c̃_t]
        X --> OG[出力ゲート<br/>o_t]

        H["前の隠れ状態<br/>h_{t-1}"] --> FG
        H --> IG
        H --> CG
        H --> OG

        C["前のセル状態<br/>c_{t-1}"] --> CM[セル更新]
        FG --> CM
        IG --> CM
        CG --> CM

        CM --> CN["新セル状態<br/>c_t"]
        CN --> HN["新隠れ状態<br/>h_t"]
        OG --> HN
    end

    style FG fill:#f5576c,color:#fff
    style IG fill:#667eea,color:#fff
    style OG fill:#43e97b,color:#000
```

### ゲートの役割

```mermaid
graph LR
    subgraph "3つのゲート"
        F["忘却ゲート f_t<br/>どの情報を忘れるか"]
        I["入力ゲート i_t<br/>どの情報を保存するか"]
        O["出力ゲート o_t<br/>どの情報を出力するか"]
    end
```

### 数式

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(c_t)$$

### 入出力仕様

| 項目 | 名前 | 形状 | 説明 |
|------|------|------|------|
| 入力 | X | [seq, batch, input] | 入力シーケンス |
| 入力 | W | [dir, 4*hidden, input] | 入力重み |
| 入力 | R | [dir, 4*hidden, hidden] | 再帰重み |
| 入力 | B | [dir, 8*hidden] | バイアス（オプション） |
| 入力 | initial_h | [dir, batch, hidden] | 初期隠れ状態（オプション） |
| 入力 | initial_c | [dir, batch, hidden] | 初期セル状態（オプション） |
| 出力 | Y | [seq, dir, batch, hidden] | 全タイムステップの出力 |
| 出力 | Y_h | [dir, batch, hidden] | 最終隠れ状態 |
| 出力 | Y_c | [dir, batch, hidden] | 最終セル状態 |

### 属性

| 属性名 | 型 | 説明 |
|--------|-----|------|
| direction | string | 'forward', 'reverse', 'bidirectional' |
| hidden_size | int | 隠れ層のサイズ |
| layout | int | 0: [seq,batch,feat], 1: [batch,seq,feat] |

### 双方向LSTM

```mermaid
graph LR
    subgraph "Bidirectional LSTM"
        X1["x_1"] --> F1[Forward]
        X2["x_2"] --> F1
        X3["x_3"] --> F1

        X1 --> B1[Backward]
        X2 --> B1
        X3 --> B1

        F1 --> C[Concat]
        B1 --> C
        C --> O["出力<br/>[seq, 2, batch, hidden]"]
    end
```

### 主な用途
- **自然言語処理**: 感情分析、品詞タグ付け
- **音声認識**
- **時系列予測**
- **機械翻訳**（エンコーダ/デコーダ）

---

## GRU（Gated Recurrent Unit）

### 説明
ゲート付き再帰ユニット。LSTMの簡略化版で、2つのゲート（リセットと更新）のみを使用します。2014年にChoらが提案。パラメータ数が少なく計算効率が良いですが、多くのタスクでLSTMと同等の性能を発揮します。

### 構造

```mermaid
graph TD
    subgraph "GRUセルの構造"
        X["入力 x_t"] --> ZG[更新ゲート<br/>z_t]
        X --> RG[リセットゲート<br/>r_t]
        X --> HG[候補隠れ状態<br/>h̃_t]

        H["前の隠れ状態<br/>h_{t-1}"] --> ZG
        H --> RG
        RG --> HG

        ZG --> HN["新隠れ状態<br/>h_t"]
        H --> HN
        HG --> HN
    end

    style ZG fill:#667eea,color:#fff
    style RG fill:#f5576c,color:#fff
```

### 数式

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$
$$\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

### 入出力仕様

| 項目 | 名前 | 形状 | 説明 |
|------|------|------|------|
| 入力 | X | [seq, batch, input] | 入力シーケンス |
| 入力 | W | [dir, 3*hidden, input] | 入力重み |
| 入力 | R | [dir, 3*hidden, hidden] | 再帰重み |
| 入力 | B | [dir, 6*hidden] | バイアス（オプション） |
| 出力 | Y | [seq, dir, batch, hidden] | 全タイムステップの出力 |
| 出力 | Y_h | [dir, batch, hidden] | 最終隠れ状態 |

### 主な用途
- **自然言語処理**
- **音声処理**
- **LSTMの軽量代替**
- **リアルタイム処理**

---

## RNN（Simple RNN）

### 説明
シンプルな再帰ニューラルネットワーク。現在の入力と前の隠れ状態を組み合わせて新しい隠れ状態を計算します。構造が単純ですが、長いシーケンスでは勾配消失問題が発生しやすいです。

### 構造

```mermaid
graph LR
    subgraph "Simple RNN"
        X["x_t"] --> T[tanh]
        H["h_{t-1}"] --> T
        T --> HN["h_t"]
    end

    style T fill:#667eea,color:#fff
```

### 数式

$$h_t = \tanh(W \cdot x_t + R \cdot h_{t-1} + b)$$

### 制限事項
- **長期依存関係の学習が困難**
- **勾配消失/爆発問題**
- **実用ではLSTM/GRUを推奨**

---

## LSTM vs GRU vs RNN 比較

```mermaid
graph TD
    subgraph "比較表"
        A["パラメータ数"]
        B["長期依存"]
        C["計算速度"]
    end

    subgraph "LSTM"
        A1["多い (4ゲート)"]
        B1["優秀"]
        C1["遅い"]
    end

    subgraph "GRU"
        A2["中程度 (3ゲート)"]
        B2["良好"]
        C2["中程度"]
    end

    subgraph "RNN"
        A3["少ない"]
        B3["弱い"]
        C3["速い"]
    end
```

### 詳細比較

| 項目 | LSTM | GRU | Simple RNN |
|------|------|-----|------------|
| ゲート数 | 3 (忘却, 入力, 出力) | 2 (更新, リセット) | 0 |
| セル状態 | あり | なし | なし |
| パラメータ | 100% | ~75% | ~25% |
| 長期依存 | 優秀 | 良好 | 弱い |
| 勾配消失 | 軽減 | 軽減 | 発生しやすい |
| 計算速度 | 遅い | 中程度 | 速い |

---

## 選択ガイド

```mermaid
graph TD
    A{タスク?}
    A -->|長いシーケンス| B[LSTM]
    A -->|中程度のシーケンス| C{リソース?}
    C -->|制限あり| D[GRU]
    C -->|十分| B
    A -->|短いシーケンス| E[GRU or Simple RNN]

    style B fill:#667eea,color:#fff
    style D fill:#f5576c,color:#fff
```

---

## データレイアウト

```mermaid
graph TD
    subgraph "layout属性"
        L0["layout=0<br/>[seq_length, batch_size, features]<br/>ONNXデフォルト"]
        L1["layout=1<br/>[batch_size, seq_length, features]<br/>PyTorchスタイル"]
    end
```
