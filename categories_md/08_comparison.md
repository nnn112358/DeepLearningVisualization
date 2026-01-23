# ONNX演算子 - 比較・論理演算 (Comparison & Logic Operations)

比較・論理演算はテンソルの要素を比較したり、ブール値に対する論理操作を行います。条件分岐やマスク生成の基盤となる重要な演算です。

## 概要図

```mermaid
graph TB
    subgraph "比較・論理演算の分類"
        A[比較・論理演算] --> B[比較演算]
        A --> C[論理演算]
        A --> D[特殊判定]

        B --> B1[Equal<br/>等しい]
        B --> B2[Greater/Less<br/>大小比較]
        B --> B3[GreaterOrEqual<br/>LessOrEqual]

        C --> C1[And<br/>論理AND]
        C --> C2[Or<br/>論理OR]
        C --> C3[Xor<br/>排他的OR]
        C --> C4[Not<br/>論理NOT]

        D --> D1[IsNaN<br/>NaN判定]
        D --> D2[IsInf<br/>無限大判定]
    end
```

---

## 比較演算

### Equal（等しい）

2つのテンソルの要素が等しいかを判定します。

```mermaid
graph LR
    subgraph "入力"
        A["A: [1, 2, 3]"]
        B["B: [1, 0, 3]"]
    end

    subgraph "Equal"
        E[A == B]
    end

    subgraph "出力"
        C["[True, False, True]"]
    end

    A --> E
    B --> E
    E --> C

    style E fill:#667eea,color:#fff
```

#### 数式
$$Y = (A == B)$$

#### 主な用途
- **条件判定**
- **マスク生成**
- **パディングトークンの検出**
- **ラベル比較**

---

### Greater / Less / GreaterOrEqual / LessOrEqual

大小関係を比較する演算子群です。

```mermaid
graph TD
    subgraph "比較演算子"
        A["Greater<br/>A > B"]
        B["Less<br/>A < B"]
        C["GreaterOrEqual<br/>A >= B"]
        D["LessOrEqual<br/>A <= B"]
    end
```

#### 使用例

```python
A = [5, 2, 3]
B = [3, 3, 3]

Greater(A, B)        # [True, False, False]
Less(A, B)           # [False, True, False]
GreaterOrEqual(A, B) # [True, False, True]
LessOrEqual(A, B)    # [False, True, True]
```

#### 主な用途
- **しきい値判定**
- **アテンションマスク生成**
- **範囲チェック**
- **ソート条件**

---

## 論理演算

### 真理値表

```mermaid
graph TD
    subgraph "論理演算の真理値表"
        direction LR
        A["AND<br/>T∧T=T<br/>T∧F=F<br/>F∧T=F<br/>F∧F=F"]
        B["OR<br/>T∨T=T<br/>T∨F=T<br/>F∨T=T<br/>F∨F=F"]
        C["XOR<br/>T⊕T=F<br/>T⊕F=T<br/>F⊕T=T<br/>F⊕F=F"]
        D["NOT<br/>¬T=F<br/>¬F=T"]
    end
```

---

### And（論理AND）

両方がTrueの場合のみTrueを返します。

```mermaid
graph LR
    subgraph "入力"
        A["A: [T, T, F, F]"]
        B["B: [T, F, T, F]"]
    end

    subgraph "And"
        AND[A ∧ B]
    end

    subgraph "出力"
        C["[T, F, F, F]"]
    end

    A --> AND
    B --> AND
    AND --> C

    style AND fill:#667eea,color:#fff
```

#### 主な用途
- **複合条件の判定**
- **マスクの組み合わせ**
- **フィルタリング条件**

---

### Or（論理OR）

どちらかがTrueならTrueを返します。

```mermaid
graph LR
    subgraph "入力"
        A["A: [T, T, F, F]"]
        B["B: [T, F, T, F]"]
    end

    subgraph "Or"
        OR[A ∨ B]
    end

    subgraph "出力"
        C["[T, T, T, F]"]
    end

    A --> OR
    B --> OR
    OR --> C

    style OR fill:#667eea,color:#fff
```

#### 主な用途
- **複合条件（いずれか）**
- **マスクの結合**
- **例外条件の追加**

---

### Xor（排他的論理和）

片方だけがTrueの場合にTrueを返します。

```mermaid
graph LR
    subgraph "入力"
        A["A: [T, T, F, F]"]
        B["B: [T, F, T, F]"]
    end

    subgraph "Xor"
        XOR[A ⊕ B]
    end

    subgraph "出力"
        C["[F, T, T, F]"]
    end

    A --> XOR
    B --> XOR
    XOR --> C

    style XOR fill:#667eea,color:#fff
```

#### 主な用途
- **差分検出**
- **異なる要素の特定**
- **トグル操作**

---

### Not（論理NOT）

ブール値を反転します。

```mermaid
graph LR
    subgraph "入力"
        X["X: [T, F, T]"]
    end

    subgraph "Not"
        NOT[¬X]
    end

    subgraph "出力"
        Y["[F, T, F]"]
    end

    X --> NOT --> Y

    style NOT fill:#f5576c,color:#fff
```

#### 主な用途
- **条件の反転**
- **マスクの反転**
- **補集合の計算**

---

## 特殊な判定演算

### IsNaN（NaN判定）

各要素がNaN（非数）かどうかを判定します。

```mermaid
graph LR
    subgraph "入力"
        X["[1.0, NaN, 3.0, NaN]"]
    end

    subgraph "IsNaN"
        N[isnan]
    end

    subgraph "出力"
        Y["[F, T, F, T]"]
    end

    X --> N --> Y

    style N fill:#667eea,color:#fff
```

#### 主な用途
- **データ検証**
- **異常値検出**
- **数値エラーのチェック**

---

### IsInf（無限大判定）

各要素が無限大（正または負）かどうかを判定します。

```mermaid
graph LR
    subgraph "入力"
        X["[1.0, -Inf, 3.0, Inf]"]
    end

    subgraph "IsInf"
        I[isinf]
    end

    subgraph "出力"
        Y["[F, T, F, T]"]
    end

    X --> I --> Y

    style I fill:#667eea,color:#fff
```

#### 属性

| 属性名 | 型 | デフォルト | 説明 |
|--------|-----|----------|------|
| detect_negative | int | 1 | 負の無限大を検出するか |
| detect_positive | int | 1 | 正の無限大を検出するか |

#### 主な用途
- **オーバーフロー検出**
- **数値安定性チェック**
- **異常値検出**

---

## Attentionマスクの生成例

```mermaid
graph TD
    subgraph "Causal Attentionマスク"
        A["位置インデックス<br/>rows: [0,1,2,3]<br/>cols: [0,1,2,3]"]
        B["Greater<br/>rows > cols"]
        C["マスク<br/>[[F,T,T,T],<br/>[F,F,T,T],<br/>[F,F,F,T],<br/>[F,F,F,F]]"]
    end

    A --> B --> C

    style B fill:#667eea,color:#fff
```

```mermaid
graph TD
    subgraph "パディングマスク"
        A["入力トークン<br/>[1, 2, 3, 0, 0]"]
        B["Equal<br/>tokens == 0"]
        C["パディングマスク<br/>[F, F, F, T, T]"]
        D["Not"]
        E["有効マスク<br/>[T, T, T, F, F]"]
    end

    A --> B --> C --> D --> E

    style B fill:#667eea,color:#fff
    style D fill:#f5576c,color:#fff
```

---

## 比較演算の入出力仕様

| 演算子 | 数式 | 入力型 | 出力型 |
|--------|------|--------|--------|
| Equal | A == B | 数値 | bool |
| Greater | A > B | 数値 | bool |
| Less | A < B | 数値 | bool |
| GreaterOrEqual | A >= B | 数値 | bool |
| LessOrEqual | A <= B | 数値 | bool |
| And | A ∧ B | bool | bool |
| Or | A ∨ B | bool | bool |
| Xor | A ⊕ B | bool | bool |
| Not | ¬A | bool | bool |
| IsNaN | isnan(X) | float | bool |
| IsInf | isinf(X) | float | bool |

---

## ブロードキャスト

全ての比較・論理演算はNumPyスタイルのブロードキャストをサポートします。

```mermaid
graph LR
    subgraph "ブロードキャスト例"
        A["A: [3, 4]<br/>B: [4]"]
        B["結果: [3, 4]"]
    end

    A --> B
```
