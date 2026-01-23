# ONNX演算子 - インデックス演算 (Index Operations)

インデックス演算はテンソルの特定の要素を選択したり、指定した位置に値を書き込んだりする演算です。埋め込みルックアップやスパーステンソルの操作に不可欠です。

## 概要図

```mermaid
graph TB
    subgraph "インデックス演算の分類"
        A[インデックス演算] --> B[収集/Gather系]
        A --> C[散布/Scatter系]
        A --> D[その他]

        B --> B1[Gather<br/>軸方向の収集]
        B --> B2[GatherElements<br/>要素単位収集]
        B --> B3[GatherND<br/>N次元収集]

        C --> C1[ScatterElements<br/>要素単位散布]
        C --> C2[ScatterND<br/>N次元散布]

        D --> D1[NonZero<br/>非ゼロインデックス]
        D --> D2[Compress<br/>条件選択]
    end
```

---

## Gather（収集）

### 説明
指定したインデックスに基づいてテンソルから要素を収集します。埋め込み層のルックアップ操作として広く使用されます。

### 動作原理

```mermaid
graph LR
    subgraph "入力データ"
        D["[[1,2],<br/>[3,4],<br/>[5,6]]<br/>(3×2)"]
    end

    subgraph "インデックス"
        I["[0, 2]"]
    end

    subgraph "Gather (axis=0)"
        G[Gather]
    end

    subgraph "出力"
        O["[[1,2],<br/>[5,6]]<br/>(2×2)"]
    end

    D --> G
    I --> G
    G --> O

    style G fill:#667eea,color:#fff
```

### 埋め込みルックアップ

```mermaid
graph TD
    subgraph "埋め込み層の動作"
        V["埋め込みテーブル<br/>[vocab_size, embed_dim]"]
        T["トークンID<br/>[batch, seq_len]"]
        G[Gather<br/>axis=0]
        E["埋め込みベクトル<br/>[batch, seq_len, embed_dim]"]
    end

    V --> G
    T --> G
    G --> E

    style G fill:#667eea,color:#fff
```

### 入出力仕様

| 項目 | 名前 | 形状 | 説明 |
|------|------|------|------|
| 入力 | data | 任意 | 入力テンソル |
| 入力 | indices | 整数テンソル | 収集するインデックス |
| 出力 | output | 計算される | 収集されたテンソル |

### 属性

| 属性名 | 型 | デフォルト | 説明 |
|--------|-----|----------|------|
| axis | int | 0 | 収集する軸 |

### 使用例

```python
# 埋め込みルックアップ
embedding = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]  # [4, 2]
indices = [1, 3, 0]
output = [[0.3, 0.4], [0.7, 0.8], [0.1, 0.2]]  # [3, 2]
```

### 主な用途
- **単語埋め込みのルックアップ**
- **インデックスによる選択**
- **バッチ内の要素選択**

---

## GatherElements（要素収集）

### 説明
インデックステンソルと同じ形状の出力を生成し、各位置で指定されたインデックスの要素を収集します。Gatherより柔軟な収集が可能です。

### 動作原理

```mermaid
graph LR
    subgraph "入力"
        D["[[1,2],<br/>[3,4]]"]
        I["[[0,0],<br/>[1,0]]"]
    end

    subgraph "axis=0"
        G[GatherElements]
    end

    subgraph "出力"
        O["[[1,2],<br/>[3,2]]"]
    end

    D --> G
    I --> G
    G --> O

    style G fill:#667eea,color:#fff
```

### 数式
```
axis=0の場合:
output[i][j][k] = input[indices[i][j][k]][j][k]
```

### 入出力仕様

| 項目 | 名前 | 形状 | 説明 |
|------|------|------|------|
| 入力 | data | 任意 | 入力テンソル |
| 入力 | indices | dataと同ランク | インデックステンソル |
| 出力 | output | indicesと同形状 | 収集されたテンソル |

### 主な用途
- **複雑なインデックス操作**
- **Top-K選択後の値取得**
- **Attentionの値選択**

---

## GatherND（N次元収集）

### 説明
N次元インデックスを使用して要素またはスライスを収集します。複数の軸にまたがるインデックス指定が可能で、より複雑な収集パターンを実現できます。

### 動作原理

```mermaid
graph LR
    subgraph "入力"
        D["[[0,1],<br/>[2,3]]"]
        I["[[0,0],<br/>[1,1]]"]
    end

    subgraph "GatherND"
        G[GatherND]
    end

    subgraph "出力"
        O["[0, 3]<br/>(data[0,0]とdata[1,1])"]
    end

    D --> G
    I --> G
    G --> O

    style G fill:#667eea,color:#fff
```

### 入出力仕様

| 項目 | 名前 | 形状 | 説明 |
|------|------|------|------|
| 入力 | data | 任意 | 入力テンソル |
| 入力 | indices | [..., index_depth] | N次元インデックス |
| 出力 | output | 計算される | 収集されたテンソル |

### 属性

| 属性名 | 型 | デフォルト | 説明 |
|--------|-----|----------|------|
| batch_dims | int | 0 | バッチ次元の数 |

### 使用例

```python
# 点群からの特徴抽出
data = [batch, H, W, C]
indices = [batch, num_points, 2]  # 各点のH,W座標
output = [batch, num_points, C]
```

### 主な用途
- **点群からの特徴抽出**
- **複雑なインデックスパターン**
- **グラフニューラルネットワーク**

---

## ScatterElements（要素散布）

### 説明
GatherElementsの逆操作。インデックスで指定した位置に値を書き込みます。更新値を指定した位置に散布します。

### 動作原理

```mermaid
graph TD
    subgraph "入力"
        D["data: [[1,2,3],<br/>[4,5,6]]"]
        I["indices: [[1,0,2],<br/>[0,2,1]]"]
        U["updates: [[10,20,30],<br/>[40,50,60]]"]
    end

    subgraph "axis=1"
        S[ScatterElements]
    end

    subgraph "出力"
        O["[[20,10,30],<br/>[40,60,50]]"]
    end

    D --> S
    I --> S
    U --> S
    S --> O

    style S fill:#f5576c,color:#fff
```

### 入出力仕様

| 項目 | 名前 | 形状 | 説明 |
|------|------|------|------|
| 入力 | data | 任意 | ベースとなるテンソル |
| 入力 | indices | 任意 | 書き込み位置のインデックス |
| 入力 | updates | indicesと同形状 | 書き込む値 |
| 出力 | output | dataと同形状 | 更新されたテンソル |

### 属性

| 属性名 | 型 | デフォルト | 説明 |
|--------|-----|----------|------|
| axis | int | 0 | 散布する軸 |
| reduction | string | 'none' | 重複時の処理 |

### reduction オプション

```mermaid
graph TD
    subgraph "重複インデックスの処理"
        A["'none'"] --> A1["最後の値で上書き"]
        B["'add'"] --> B1["値を加算"]
        C["'mul'"] --> C1["値を乗算"]
        D["'max'"] --> D1["最大値を採用"]
        E["'min'"] --> E1["最小値を採用"]
    end
```

### 主な用途
- **One-hotエンコーディングの作成**
- **スパース更新**
- **勾配の散布**

---

## ScatterND（N次元散布）

### 説明
GatherNDの逆操作。N次元インデックスで指定した位置に値を書き込みます。

### 動作原理

```mermaid
graph LR
    subgraph "入力"
        D["[1,2,3,4,5,6,7,8]"]
        I["[[4],[3],[1],[7]]"]
        U["[9,10,11,12]"]
    end

    subgraph "ScatterND"
        S[ScatterND]
    end

    subgraph "出力"
        O["[1,11,3,10,9,6,7,12]"]
    end

    D --> S
    I --> S
    U --> S
    S --> O

    style S fill:#f5576c,color:#fff
```

### 主な用途
- **スパーステンソルの構築**
- **点群の特徴書き込み**
- **グラフ更新**

---

## NonZero（非ゼロインデックス）

### 説明
入力テンソルで0でない要素のインデックスを返します。スパースデータの処理やマスク操作に使用されます。

### 動作原理

```mermaid
graph LR
    subgraph "入力"
        X["[[1,0],<br/>[0,2]]"]
    end

    subgraph "NonZero"
        N[NonZero]
    end

    subgraph "出力"
        O["[[0,1],<br/>[0,1]]<br/>(位置(0,0)と(1,1))"]
    end

    X --> N --> O

    style N fill:#667eea,color:#fff
```

### 入出力仕様

| 項目 | 名前 | 形状 | 説明 |
|------|------|------|------|
| 入力 | X | 任意 | 入力テンソル |
| 出力 | Y | [rank, num_nonzero] | 非ゼロ要素のインデックス |

### 主な用途
- **スパースデータ処理**
- **マスクの適用**
- **条件に合う要素の検索**

---

## Compress（圧縮選択）

### 説明
ブール条件に基づいて要素を選択します。条件がTrueの要素のみを抽出して返します。

### 動作原理

```mermaid
graph LR
    subgraph "入力"
        I["[[1,2],<br/>[3,4],<br/>[5,6]]"]
        C["[False,True,True]"]
    end

    subgraph "axis=0"
        CP[Compress]
    end

    subgraph "出力"
        O["[[3,4],<br/>[5,6]]"]
    end

    I --> CP
    C --> CP
    CP --> O

    style CP fill:#667eea,color:#fff
```

### 入出力仕様

| 項目 | 名前 | 形状 | 説明 |
|------|------|------|------|
| 入力 | input | 任意 | 入力テンソル |
| 入力 | condition | ブール | 条件テンソル |
| 出力 | output | 圧縮後 | 選択された要素 |

### 属性

| 属性名 | 型 | デフォルト | 説明 |
|--------|-----|----------|------|
| axis | int | None | 圧縮する軸（省略時は平坦化） |

### 主な用途
- **条件フィルタリング**
- **マスク適用**
- **有効なデータの抽出**

---

## Gather vs GatherElements vs GatherND

```mermaid
graph TD
    subgraph "使い分け"
        A["Gather"] --> A1["1つの軸に沿って収集<br/>埋め込みルックアップに最適"]
        B["GatherElements"] --> B1["要素単位で収集<br/>出力形状=インデックス形状"]
        C["GatherND"] --> C1["複数軸の座標で収集<br/>点群など複雑なパターン"]
    end
```

### 比較表

| 演算子 | インデックス | 出力形状 | 用途 |
|--------|------------|---------|------|
| Gather | 1軸方向 | indices形状 + data残り次元 | 埋め込み |
| GatherElements | 要素単位 | indices形状 | Top-K値取得 |
| GatherND | N次元座標 | 座標に依存 | 点群、グラフ |
