# ONNX演算子 - 行列演算 (Matrix Operations)

行列演算はニューラルネットワークの全結合層やAttention機構の基盤となる演算です。線形変換と行列計算を提供します。

## 概要図

```mermaid
graph TB
    subgraph "行列演算の分類"
        A[行列演算] --> B[行列乗算]
        A --> C[量子化行列演算]
        A --> D[行列変換]

        B --> B1[MatMul<br/>標準行列乗算]
        B --> B2[Gemm<br/>一般行列乗算]

        C --> C1[MatMulInteger<br/>整数行列乗算]
        C --> C2[QLinearMatMul<br/>量子化行列乗算]

        D --> D1[Trilu<br/>三角行列]
    end
```

---

## MatMul（行列乗算）

### 説明
2つのテンソルの行列乗算を行います。ニューラルネットワークの全結合層やAttention機構の基本演算です。高次元テンソルではバッチ行列乗算として動作します。

### 動作原理

```mermaid
graph LR
    subgraph "2D行列乗算"
        A["A: [3, 4]"]
        B["B: [4, 5]"]
        M[MatMul]
        C["C: [3, 5]"]
    end

    A --> M
    B --> M
    M --> C

    style M fill:#667eea,color:#fff
```

### 数式
$$C_{ij} = \sum_{k} A_{ik} \times B_{kj}$$

### サイズ制約

```mermaid
graph TD
    subgraph "次元の制約"
        A["A: [..., M, K]"]
        B["B: [..., K, N]"]
        C["C: [..., M, N]"]
        D["Aの最後の次元 = Bの最後から2番目の次元"]
    end

    A --> D
    B --> D
    D --> C
```

### バッチ行列乗算

```mermaid
graph LR
    subgraph "バッチ処理"
        A["A: [batch, 3, 4]"]
        B["B: [batch, 4, 5]"]
        M[MatMul]
        C["C: [batch, 3, 5]"]
    end

    A --> M
    B --> M
    M --> C

    style M fill:#667eea,color:#fff
```

### Attention での使用

```mermaid
graph TD
    subgraph "Self-Attention"
        Q["Q: [batch, heads, seq, dim]"]
        K["K: [batch, heads, dim, seq]<br/>(転置後)"]
        S["Scores = Q @ K<br/>[batch, heads, seq, seq]"]
        V["V: [batch, heads, seq, dim]"]
        O["Output = Softmax(Scores) @ V<br/>[batch, heads, seq, dim]"]
    end

    Q --> S
    K --> S
    S --> O
    V --> O

    style S fill:#667eea,color:#fff
    style O fill:#667eea,color:#fff
```

### 入出力仕様

| 項目 | 名前 | 形状 | 説明 |
|------|------|------|------|
| 入力 | A | [..., M, K] | 第1テンソル |
| 入力 | B | [..., K, N] | 第2テンソル |
| 出力 | Y | [..., M, N] | 結果テンソル |

### 主な用途
- **全結合層**
- **Attention (Q @ K^T, scores @ V)**
- **座標変換**
- **特徴変換**

---

## Gemm（一般行列乗算）

### 説明
General Matrix Multiplication。行列乗算にスケーリングとバイアス加算を組み合わせた演算です。全結合層を効率的に1つの演算子で表現できます。

### 数式
$$Y = \alpha \cdot A' \times B' + \beta \cdot C$$

$$A' = \begin{cases} A^T & \text{if transA} \\ A & \text{otherwise} \end{cases}$$

$$B' = \begin{cases} B^T & \text{if transB} \\ B & \text{otherwise} \end{cases}$$

### 全結合層としての使用

```mermaid
graph LR
    subgraph "Linear Layer: Y = X @ W^T + b"
        X["X: [batch, in_features]"]
        W["W: [out_features, in_features]"]
        b["b: [out_features]"]
        G["Gemm<br/>transB=1"]
        Y["Y: [batch, out_features]"]
    end

    X --> G
    W --> G
    b --> G
    G --> Y

    style G fill:#667eea,color:#fff
```

### 入出力仕様

| 項目 | 名前 | 形状 | 説明 |
|------|------|------|------|
| 入力 | A | [M, K] or [K, M] | 入力テンソル |
| 入力 | B | [K, N] or [N, K] | 重みテンソル |
| 入力 | C | [M, N] or [N] | バイアス（オプション） |
| 出力 | Y | [M, N] | 結果テンソル |

### 属性

| 属性名 | 型 | デフォルト | 説明 |
|--------|-----|----------|------|
| alpha | float | 1.0 | A×Bの係数 |
| beta | float | 1.0 | Cの係数 |
| transA | int | 0 | Aを転置するか |
| transB | int | 0 | Bを転置するか |

### MatMul vs Gemm

| 特性 | MatMul | Gemm |
|------|--------|------|
| バイアス | 別途Add必要 | 組み込み |
| スケーリング | なし | alpha, beta |
| 転置 | 別途Transpose必要 | 組み込み |
| 次元 | 任意次元 | 2次元のみ |

---

## MatMulInteger（整数行列乗算）

### 説明
整数型での行列乗算を行います。量子化されたモデルで使用され、int8やuint8の入力をサポートします。

### 数式
$$Y = (A - a\_zero\_point) \times (B - b\_zero\_point)$$

### 動作フロー

```mermaid
graph TD
    subgraph "量子化行列乗算"
        A["A: int8 [M, K]"]
        AZ["a_zero_point"]
        B["B: int8 [K, N]"]
        BZ["b_zero_point"]
        S["(A - AZ) @ (B - BZ)"]
        Y["Y: int32 [M, N]"]
    end

    A --> S
    AZ --> S
    B --> S
    BZ --> S
    S --> Y

    style S fill:#667eea,color:#fff
```

### 入出力仕様

| 項目 | 名前 | 型 | 説明 |
|------|------|-----|------|
| 入力 | A | int8/uint8 | 第1テンソル |
| 入力 | B | int8/uint8 | 第2テンソル |
| 入力 | a_zero_point | int8/uint8 | Aのゼロポイント（オプション） |
| 入力 | b_zero_point | int8/uint8 | Bのゼロポイント（オプション） |
| 出力 | Y | int32 | 結果（累積用にint32） |

### 主な用途
- **量子化推論**
- **エッジデバイス向け最適化**
- **高速推論**

---

## QLinearMatMul（量子化線形行列乗算）

### 説明
完全な量子化パイプラインでの行列乗算。入力、重み、出力それぞれにスケールとゼロポイントを持ちます。

### 数式
$$y = \frac{(a\_scale \times (a - a\_zp)) \times (b\_scale \times (b - b\_zp))}{y\_scale} + y\_zp$$

### 量子化パイプライン

```mermaid
graph TD
    subgraph "QLinearMatMul"
        A["a (quantized)"]
        AS["a_scale, a_zero_point"]
        B["b (quantized)"]
        BS["b_scale, b_zero_point"]
        D["逆量子化 → 計算 → 再量子化"]
        YS["y_scale, y_zero_point"]
        Y["y (quantized)"]
    end

    A --> D
    AS --> D
    B --> D
    BS --> D
    D --> Y
    YS --> D

    style D fill:#667eea,color:#fff
```

### 主な用途
- **量子化推論パイプライン**
- **INT8推論**
- **モデル軽量化**

---

## Trilu（三角行列）

### 説明
上三角行列または下三角行列を抽出します。Attentionマスクの生成などに使用されます。

### 動作原理

```mermaid
graph TD
    subgraph "Trilu の動作"
        I["入力:<br/>[[1,2,3],<br/>[4,5,6],<br/>[7,8,9]]"]
        U["upper=1 (上三角):<br/>[[1,2,3],<br/>[0,5,6],<br/>[0,0,9]]"]
        L["upper=0 (下三角):<br/>[[1,0,0],<br/>[4,5,0],<br/>[7,8,9]]"]
    end

    I --> U
    I --> L
```

### 対角オフセット

```mermaid
graph TD
    subgraph "k（対角オフセット）の効果"
        A["k=0: 主対角線"]
        B["k=1: 主対角線の1つ上"]
        C["k=-1: 主対角線の1つ下"]
    end
```

### 属性

| 属性名 | 型 | デフォルト | 説明 |
|--------|-----|----------|------|
| upper | int | 1 | 上三角(1)か下三角(0)か |

### 入力

| 名前 | 説明 |
|------|------|
| input | 入力テンソル [..., M, N] |
| k | 対角オフセット（オプション、デフォルト: 0） |

### Causal Attention マスク

```mermaid
graph LR
    subgraph "Causal Mask生成"
        A["ones(seq, seq)"]
        T["Trilu<br/>upper=1"]
        M["マスク<br/>[[1,1,1,1],<br/>[0,1,1,1],<br/>[0,0,1,1],<br/>[0,0,0,1]]"]
    end

    A --> T --> M

    style T fill:#f5576c,color:#fff
```

### 主な用途
- **Causal Attention マスク**
- **三角分解**
- **時系列マスキング**

---

## 行列演算の選択ガイド

```mermaid
graph TD
    A{用途?}
    A -->|高次元バッチ| B[MatMul]
    A -->|FC層| C[Gemm]
    A -->|量子化| D{精度?}
    D -->|int出力| E[MatMulInteger]
    D -->|完全量子化| F[QLinearMatMul]

    style B fill:#667eea,color:#fff
    style C fill:#f5576c,color:#fff
    style E fill:#43e97b,color:#000
    style F fill:#fa709a,color:#fff
```

### 比較表

| 演算子 | 次元 | バイアス | 量子化 | 用途 |
|--------|------|---------|--------|------|
| MatMul | 任意 | なし | なし | Attention |
| Gemm | 2D | あり | なし | FC層 |
| MatMulInteger | 2D | なし | int8/uint8 | 量子化推論 |
| QLinearMatMul | 2D | なし | 完全 | INT8パイプライン |
