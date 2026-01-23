# ONNX演算子 - 数学関数 (Math Functions)

数学関数は要素ごとの数学的変換を提供します。指数・対数関数、三角関数、丸め関数など、科学計算や確率計算の基盤となる演算です。

## 概要図

```mermaid
graph TB
    subgraph "数学関数の分類"
        A[数学関数] --> B[指数・対数]
        A --> C[丸め関数]
        A --> D[三角関数]
        A --> E[その他]

        B --> B1[Exp<br/>指数関数]
        B --> B2[Log<br/>自然対数]

        C --> C1[Ceil<br/>切り上げ]
        C --> C2[Floor<br/>切り捨て]
        C --> C3[Round<br/>四捨五入]
        C --> C4[Clip<br/>クリッピング]

        D --> D1[Sin/Cos/Tan]
        D --> D2[Asin/Acos/Atan]
        D --> D3[Sinh/Cosh/Tanh]

        E --> E1[Neg<br/>符号反転]
        E --> E2[Sign<br/>符号関数]
        E --> E3[Erf<br/>誤差関数]
    end
```

---

## Exp（指数関数）

### 説明
各要素の指数関数（eのx乗）を計算します。Softmax、確率計算、正規分布などで広く使用されます。

### 動作原理

```mermaid
graph LR
    subgraph "入力"
        X["[0, 1, 2]"]
    end

    subgraph "Exp"
        E[e^x]
    end

    subgraph "出力"
        Y["[1.0, 2.718, 7.389]"]
    end

    X --> E --> Y

    style E fill:#667eea,color:#fff
```

### 数式
$$Y = e^X$$

### 注意点

```mermaid
graph TD
    A["大きな入力値"] --> B["exp(100) → オーバーフロー"]
    C["Softmaxでの対策"] --> D["max(X)を引いてからexp"]
```

### 主な用途
- **Softmax計算**
- **確率分布**
- **指数関数的成長/減衰**

---

## Log（自然対数）

### 説明
各要素の自然対数（底がeの対数）を計算します。損失関数、エントロピー計算などで使用されます。

### 動作原理

```mermaid
graph LR
    subgraph "入力"
        X["[1, 2.718, 7.389]"]
    end

    subgraph "Log"
        L[ln(x)]
    end

    subgraph "出力"
        Y["[0, 1.0, 2.0]"]
    end

    X --> L --> Y

    style L fill:#667eea,color:#fff
```

### 数式
$$Y = \ln(X)$$

### 注意点
- 入力は**正の値**である必要があります
- 0以下の値でNaNまたは-Infが発生

### 主な用途
- **CrossEntropy損失**
- **情報エントロピー**
- **対数スケール変換**

---

## 丸め関数

### Ceil（切り上げ）

値以上の最小の整数を返します。

```mermaid
graph LR
    subgraph "入力"
        X["[1.1, 2.5, -1.7, -2.3]"]
    end

    subgraph "Ceil"
        C["⌈x⌉"]
    end

    subgraph "出力"
        Y["[2, 3, -1, -2]"]
    end

    X --> C --> Y

    style C fill:#667eea,color:#fff
```

### Floor（切り捨て）

値以下の最大の整数を返します。

```mermaid
graph LR
    subgraph "入力"
        X["[1.9, 2.5, -1.3, -2.7]"]
    end

    subgraph "Floor"
        F["⌊x⌋"]
    end

    subgraph "出力"
        Y["[1, 2, -2, -3]"]
    end

    X --> F --> Y

    style F fill:#667eea,color:#fff
```

### Round（四捨五入）

最も近い整数に丸めます。0.5の場合は偶数方向に丸めます（銀行家の丸め）。

```mermaid
graph LR
    subgraph "入力"
        X["[0.4, 0.5, 0.6, 1.5, 2.5]"]
    end

    subgraph "Round"
        R["round"]
    end

    subgraph "出力"
        Y["[0, 0, 1, 2, 2]<br/>(0.5は偶数方向)"]
    end

    X --> R --> Y

    style R fill:#667eea,color:#fff
```

### 丸め関数の比較

| 入力 | Ceil | Floor | Round |
|------|------|-------|-------|
| 1.5 | 2 | 1 | 2 |
| 2.5 | 3 | 2 | 2 |
| -1.5 | -1 | -2 | -2 |
| -2.5 | -2 | -3 | -2 |

---

## Clip（クリッピング）

### 説明
値を指定した最小値と最大値の範囲内に制限します。

### 動作原理

```mermaid
graph LR
    subgraph "入力"
        X["[-5, 0, 5, 10, 15]"]
    end

    subgraph "Clip (min=0, max=10)"
        C[Clip]
    end

    subgraph "出力"
        Y["[0, 0, 5, 10, 10]"]
    end

    X --> C --> Y

    style C fill:#667eea,color:#fff
```

### 数式
$$Y = \min(\max(X, min\_val), max\_val)$$

### ReLU6の実装

```mermaid
graph LR
    A["入力"] --> B["Clip<br/>min=0, max=6"] --> C["ReLU6出力"]

    style B fill:#f5576c,color:#fff
```

### 主な用途
- **勾配クリッピング**
- **値の正規化**
- **ReLU6の実装**

---

## Neg / Sign（符号演算）

### Neg（符号反転）

```mermaid
graph LR
    subgraph "入力"
        X["[1, -2, 0, 3]"]
    end

    subgraph "Neg"
        N["-x"]
    end

    subgraph "出力"
        Y["[-1, 2, 0, -3]"]
    end

    X --> N --> Y

    style N fill:#667eea,color:#fff
```

### Sign（符号関数）

```mermaid
graph LR
    subgraph "入力"
        X["[3, -2, 0, -0.5, 7]"]
    end

    subgraph "Sign"
        S["sign(x)"]
    end

    subgraph "出力"
        Y["[1, -1, 0, -1, 1]"]
    end

    X --> S --> Y

    style S fill:#667eea,color:#fff
```

### Sign の定義

$$\text{sign}(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x = 0 \\ -1 & \text{if } x < 0 \end{cases}$$

---

## Erf（誤差関数）

### 説明
ガウス誤差関数を計算します。GELU活性化関数の計算や統計処理で使用されます。

### 数式
$$\text{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt$$

### グラフの特性

```mermaid
graph TD
    subgraph "Erfの特性"
        A["erf(-∞) = -1"]
        B["erf(0) = 0"]
        C["erf(+∞) = 1"]
        D["出力範囲: [-1, 1]"]
    end
```

### GELUでの使用

```mermaid
graph LR
    X["入力 x"] --> E["x * Φ(x)<br/>Φ(x) = 0.5(1 + erf(x/√2))"]
    E --> Y["GELU出力"]
```

### 主な用途
- **GELU活性化関数**
- **正規分布の累積分布**
- **統計的信頼区間**

---

## 三角関数・双曲線関数

### 基本三角関数

```mermaid
graph TD
    subgraph "三角関数"
        A["Sin(x)"] --> A1["正弦関数"]
        B["Cos(x)"] --> B1["余弦関数"]
        C["Tan(x)"] --> C1["正接関数<br/>sin(x)/cos(x)"]
    end
```

### 逆三角関数

| 関数 | 数式 | 出力範囲 |
|------|------|----------|
| Asin | arcsin(x) | [-π/2, π/2] |
| Acos | arccos(x) | [0, π] |
| Atan | arctan(x) | [-π/2, π/2] |

### 双曲線関数

$$\cosh(x) = \frac{e^x + e^{-x}}{2}$$
$$\sinh(x) = \frac{e^x - e^{-x}}{2}$$
$$\tanh(x) = \frac{\sinh(x)}{\cosh(x)} = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

### 主な用途
- **位置エンコーディング**（Transformer）
- **角度計算**
- **座標変換**
- **信号処理**

---

## Einsum（アインシュタイン縮約）

### 説明
アインシュタイン縮約記法を使用して、柔軟なテンソル演算を行います。行列乗算、転置、縮約、外積などを統一的に表現できます。

### 記法の例

```mermaid
graph TD
    subgraph "Einsum記法"
        A["'ij,jk->ik'"] --> A1["行列乗算<br/>C[i,k] = Σⱼ A[i,j] × B[j,k]"]
        B["'ij->ji'"] --> B1["転置"]
        C["'ii->'"] --> C1["トレース"]
        D["'bij,bjk->bik'"] --> D1["バッチ行列乗算"]
    end
```

### 使用例

```python
# 行列乗算
equation = "ij,jk->ik"
A: [2, 3]
B: [3, 4]
C: [2, 4]

# Attention: scores = Q @ K^T
equation = "bhqd,bhkd->bhqk"
Q: [batch, heads, q_len, dim]
K: [batch, heads, k_len, dim]
scores: [batch, heads, q_len, k_len]
```

### 主な用途
- **複雑なテンソル演算**
- **Attention機構**
- **テンソルネットワーク**

---

## 数学関数の一覧

| 関数 | 数式 | 説明 |
|------|------|------|
| Exp | e^x | 指数関数 |
| Log | ln(x) | 自然対数 |
| Ceil | ⌈x⌉ | 切り上げ |
| Floor | ⌊x⌋ | 切り捨て |
| Round | round(x) | 四捨五入 |
| Clip | clip(x, min, max) | クリッピング |
| Neg | -x | 符号反転 |
| Sign | sign(x) | 符号関数 |
| Erf | erf(x) | 誤差関数 |
| Sin/Cos/Tan | 三角関数 | 周期関数 |
| Sinh/Cosh/Tanh | 双曲線関数 | 指数ベース |
