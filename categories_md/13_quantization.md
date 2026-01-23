# ONNX演算子 - 量子化演算 (Quantization Operations)

量子化演算は浮動小数点数を整数に変換し、モデルの軽量化と高速化を実現します。エッジデバイスへのデプロイやメモリ効率の向上に不可欠です。

## 概要図

```mermaid
graph TB
    subgraph "量子化演算の分類"
        A[量子化演算] --> B[量子化]
        A --> C[逆量子化]
        A --> D[量子化演算]

        B --> B1[QuantizeLinear<br/>静的量子化]
        B --> B2[DynamicQuantizeLinear<br/>動的量子化]

        C --> C1[DequantizeLinear<br/>逆量子化]

        D --> D1[QLinearConv<br/>量子化畳み込み]
        D --> D2[QLinearMatMul<br/>量子化行列乗算]
    end
```

---

## 量子化の基本概念

### スケールとゼロポイント

```mermaid
graph LR
    subgraph "量子化と逆量子化"
        R["実数値 r"]
        Q["整数値 q"]

        R -->|"量子化<br/>q = round(r/scale) + zp"| Q
        Q -->|"逆量子化<br/>r = scale × (q - zp)"| R
    end
```

### 量子化の種類

```mermaid
graph TD
    subgraph "量子化方式"
        A["対称量子化<br/>zero_point = 0"]
        B["非対称量子化<br/>zero_point ≠ 0"]
    end

    subgraph "粒度"
        C["Per-tensor<br/>テンソル全体で1つのスケール"]
        D["Per-channel<br/>チャネルごとに異なるスケール"]
    end
```

### 量子化による圧縮

| 元の型 | 量子化後 | 圧縮率 |
|--------|---------|--------|
| FP32 | INT8 | 4x |
| FP32 | INT4 | 8x |
| FP16 | INT8 | 2x |

---

## QuantizeLinear（線形量子化）

### 説明
浮動小数点数を整数に量子化します。モデルの軽量化と高速化のために使用されます。

### 動作原理

```mermaid
graph LR
    subgraph "入力"
        X["float: [0.0, 0.5, 1.0, 1.5]"]
    end

    subgraph "量子化"
        Q["scale=0.5<br/>zp=0"]
    end

    subgraph "出力"
        Y["int8: [0, 1, 2, 3]"]
    end

    X --> Q --> Y

    style Q fill:#667eea,color:#fff
```

### 数式
$$y = \text{saturate}(\text{round}(x / \text{scale}) + \text{zero\_point})$$

saturateは出力型の範囲にクリップ:
- int8: [-128, 127]
- uint8: [0, 255]

### 入出力仕様

| 項目 | 名前 | 型 | 説明 |
|------|------|-----|------|
| 入力 | x | float | 浮動小数点テンソル |
| 入力 | y_scale | float | スケール係数 |
| 入力 | y_zero_point | int8/uint8 | ゼロポイント（オプション） |
| 出力 | y | int8/uint8 | 量子化テンソル |

### 属性

| 属性名 | 型 | 説明 |
|--------|-----|------|
| axis | int | per-channel量子化の軸 |
| saturate | int | 飽和するか（デフォルト: 1） |

### 主な用途
- **モデル圧縮**: FP32 → INT8で約4倍削減
- **推論高速化**
- **エッジデバイスへのデプロイ**

---

## DequantizeLinear（線形逆量子化）

### 説明
量子化された整数を浮動小数点数に戻します。

### 動作原理

```mermaid
graph LR
    subgraph "入力"
        X["int8: [0, 1, 2, 3]"]
    end

    subgraph "逆量子化"
        D["scale=0.5<br/>zp=0"]
    end

    subgraph "出力"
        Y["float: [0.0, 0.5, 1.0, 1.5]"]
    end

    X --> D --> Y

    style D fill:#f5576c,color:#fff
```

### 数式
$$y = (x - \text{zero\_point}) \times \text{scale}$$

### 主な用途
- **量子化演算の出力変換**
- **混合精度推論**
- **精度の確認**

---

## DynamicQuantizeLinear（動的線形量子化）

### 説明
入力の最小/最大値からスケールとゼロポイントを動的に計算し、量子化を行います。事前のキャリブレーションが不要です。

### 動作フロー

```mermaid
graph TD
    subgraph "動的量子化"
        X["入力 x"]
        MM["min/max 計算"]
        SC["scale = (max-min)/(qmax-qmin)"]
        ZP["zero_point 計算"]
        Q["量子化"]
        Y["出力 y, scale, zp"]
    end

    X --> MM --> SC --> ZP --> Q --> Y

    style Q fill:#667eea,color:#fff
```

### 出力

| 名前 | 型 | 説明 |
|------|-----|------|
| y | uint8 | 量子化テンソル |
| y_scale | float | 計算されたスケール |
| y_zero_point | uint8 | 計算されたゼロポイント |

### 主な用途
- **キャリブレーション不要の量子化**
- **動的な入力範囲への対応**
- **プロトタイピング**

---

## QLinearConv（量子化線形畳み込み）

### 説明
量子化された入力と重みを使用して畳み込みを行います。INT8での効率的な畳み込み演算を実現します。

### 動作フロー

```mermaid
graph TD
    subgraph "QLinearConv"
        X["x (int8)"]
        XS["x_scale, x_zp"]
        W["w (int8)"]
        WS["w_scale, w_zp"]
        YS["y_scale, y_zp"]

        D1["逆量子化"]
        C["Conv"]
        Q1["量子化"]
        Y["y (int8)"]
    end

    X --> D1
    XS --> D1
    W --> D1
    WS --> D1
    D1 --> C
    C --> Q1
    YS --> Q1
    Q1 --> Y

    style C fill:#667eea,color:#fff
```

### 入出力仕様

| 項目 | 名前 | 説明 |
|------|------|------|
| 入力 | x | 量子化入力 |
| 入力 | x_scale, x_zero_point | 入力の量子化パラメータ |
| 入力 | w | 量子化重み |
| 入力 | w_scale, w_zero_point | 重みの量子化パラメータ |
| 入力 | y_scale, y_zero_point | 出力の量子化パラメータ |
| 入力 | B | バイアス（int32、オプション） |
| 出力 | y | 量子化出力 |

### 主な用途
- **量子化されたCNN**
- **INT8畳み込み**
- **モバイル/エッジ推論**

---

## 量子化パイプライン

```mermaid
graph LR
    subgraph "完全量子化推論"
        A["入力<br/>(float)"]
        Q1["QuantizeLinear"]
        B["QLinearConv"]
        C["QLinearConv"]
        D["QLinearMatMul"]
        DQ["DequantizeLinear"]
        E["出力<br/>(float)"]
    end

    A --> Q1 --> B --> C --> D --> DQ --> E

    style Q1 fill:#667eea,color:#fff
    style DQ fill:#f5576c,color:#fff
```

---

## キャリブレーション方法

```mermaid
graph TD
    subgraph "スケール決定方法"
        A["Min-Max<br/>最小/最大値を使用"]
        B["Percentile<br/>外れ値を除外"]
        C["MSE<br/>量子化誤差を最小化"]
        D["Entropy<br/>KLダイバージェンスを最小化"]
    end
```

### 各方法の特徴

| 方法 | 特徴 | 用途 |
|------|------|------|
| Min-Max | シンプル、外れ値に敏感 | 基本的な量子化 |
| Percentile | 外れ値に頑健 | 一般的な推論 |
| MSE | 精度重視 | 精度が重要な場合 |
| Entropy | TensorRTで使用 | 高精度量子化 |

---

## 量子化対応レイヤー

```mermaid
graph TD
    subgraph "量子化しやすいレイヤー"
        A["Conv / ConvTranspose"]
        B["MatMul / Gemm"]
        C["ReLU (無料)"]
    end

    subgraph "注意が必要なレイヤー"
        D["Sigmoid / Tanh"]
        E["Softmax"]
        F["LayerNorm"]
    end
```

---

## 精度への影響

```mermaid
graph LR
    subgraph "量子化の精度影響"
        A["FP32<br/>基準精度"]
        B["INT8<br/>~0.1-1%低下"]
        C["INT4<br/>~1-5%低下"]
    end

    A --> B --> C
```

### 精度低下を抑える方法
1. **Per-channel量子化**: より細かいスケール制御
2. **混合精度**: 敏感なレイヤーはFP16/FP32で維持
3. **Quantization-Aware Training (QAT)**: 訓練時に量子化をシミュレート
4. **適切なキャリブレーション**: 代表的なデータで統計を取得
