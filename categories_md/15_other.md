# ONNX演算子 - その他の演算 (Other Operations)

その他の演算はTopK選択、条件分岐、物体検出のNMS、Transformerのアテンション、ビット演算など、特殊な用途に使用される演算子を含みます。

## 概要図

```mermaid
graph TB
    subgraph "その他の演算の分類"
        A[その他] --> B[選択・条件]
        A --> C[正則化]
        A --> D[画像処理]
        A --> E[Transformer]
        A --> F[ビット演算]

        B --> B1[TopK<br/>上位K選択]
        B --> B2[Where<br/>条件選択]
        B --> B3[If<br/>条件分岐]
        B --> B4[NonMaxSuppression<br/>NMS]

        C --> C1[Dropout<br/>ドロップアウト]

        D --> D1[GridSample<br/>グリッドサンプリング]
        D --> D2[RoiAlign<br/>ROIアライン]

        E --> E1[Attention<br/>アテンション]
        E --> E2[RotaryEmbedding<br/>RoPE]

        F --> F1[BitShift<br/>ビットシフト]
        F --> F2[Bitwise演算]
    end
```

---

## TopK（上位K選択）

### 説明
指定した軸に沿って上位K個（または下位K個）の値とそのインデックスを返します。

### 動作原理

```mermaid
graph LR
    subgraph "入力"
        X["[3, 1, 4, 1, 5, 9, 2, 6]"]
        K["K=3"]
    end

    subgraph "TopK"
        T[TopK<br/>largest=1]
    end

    subgraph "出力"
        V["Values: [9, 6, 5]"]
        I["Indices: [5, 7, 4]"]
    end

    X --> T
    K --> T
    T --> V
    T --> I

    style T fill:#667eea,color:#fff
```

### 属性

| 属性名 | 型 | デフォルト | 説明 |
|--------|-----|----------|------|
| axis | int | -1 | 検索する軸 |
| largest | int | 1 | 最大値を選ぶか |
| sorted | int | 1 | 結果をソートするか |

### 主な用途
- **分類のTop-K予測**
- **ビームサーチ**
- **k-NN検索**
- **物体検出のNMS前処理**

---

## Where（条件選択）

### 説明
条件テンソルに基づいて、2つのテンソルから要素を選択します。三項演算子のテンソル版です。

### 動作原理

```mermaid
graph LR
    subgraph "入力"
        C["condition: [T, F, T, F]"]
        X["X: [1, 2, 3, 4]"]
        Y["Y: [10, 20, 30, 40]"]
    end

    subgraph "Where"
        W["condition ? X : Y"]
    end

    subgraph "出力"
        O["[1, 20, 3, 40]"]
    end

    C --> W
    X --> W
    Y --> W
    W --> O

    style W fill:#667eea,color:#fff
```

### 数式
$$\text{output}[i] = X[i] \text{ if condition}[i] \text{ else } Y[i]$$

### 主な用途
- **条件付き値の選択**
- **マスク適用**
- **クリッピング**
- **NaN/Infの置換**

---

## If（条件分岐）

### 説明
条件に基づいて異なるサブグラフを実行します。動的なモデル構造を可能にします。

### 動作フロー

```mermaid
graph TD
    subgraph "If演算"
        C["条件 cond"]
        C -->|True| T["then_branch"]
        C -->|False| E["else_branch"]
        T --> O["outputs"]
        E --> O
    end
```

### 属性

| 属性名 | 型 | 説明 |
|--------|-----|------|
| then_branch | GraphProto | 条件がTrueの場合のサブグラフ |
| else_branch | GraphProto | 条件がFalseの場合のサブグラフ |

### 主な用途
- **条件付き処理**
- **早期終了**
- **動的アーキテクチャ**

---

## NonMaxSuppression（非最大抑制）

### 説明
物体検出で重複するバウンディングボックスを除去します。IoU（Intersection over Union）に基づいて、重複の少ない代表的な検出結果のみを保持します。

### 動作原理

```mermaid
graph TD
    subgraph "NMS処理"
        A["多数の検出ボックス"]
        B["スコア順にソート"]
        C["最高スコアを選択"]
        D["IoU > threshold の<br/>ボックスを削除"]
        E["次のボックスを選択"]
        F["代表的なボックスのみ残る"]
    end

    A --> B --> C --> D --> E
    E -->|残りあり| D
    E -->|完了| F

    style C fill:#667eea,color:#fff
    style D fill:#f5576c,color:#fff
```

### 入出力仕様

| 項目 | 名前 | 形状 | 説明 |
|------|------|------|------|
| 入力 | boxes | [batch, num_boxes, 4] | バウンディングボックス |
| 入力 | scores | [batch, num_classes, num_boxes] | スコア |
| 入力 | max_output_boxes_per_class | スカラー | クラスあたり最大出力数 |
| 入力 | iou_threshold | スカラー | IoU閾値 |
| 入力 | score_threshold | スカラー | スコア閾値 |
| 出力 | selected_indices | [num_selected, 3] | 選択されたインデックス |

### 主な用途
- **物体検出**: YOLO, SSD, Faster R-CNN
- **インスタンスセグメンテーション**
- **キーポイント検出**

---

## Dropout（ドロップアウト）

### 説明
訓練時にランダムに要素を0にすることで過学習を防ぎます。推論時は全ての要素を使用します。

### 動作原理

```mermaid
graph LR
    subgraph "訓練時"
        X1["[1, 2, 3, 4, 5]"]
        D1["Dropout<br/>ratio=0.4"]
        Y1["[1.67, 0, 5, 6.67, 0]"]
    end

    subgraph "推論時"
        X2["[1, 2, 3, 4, 5]"]
        D2["Dropout<br/>(通過)"]
        Y2["[1, 2, 3, 4, 5]"]
    end

    X1 --> D1 --> Y1
    X2 --> D2 --> Y2

    style D1 fill:#667eea,color:#fff
```

### 数式
$$Y = X \times \text{mask} / (1 - \text{ratio})$$

### 主な用途
- **過学習防止**
- **アンサンブル効果**
- **正則化**

---

## GridSample（グリッドサンプリング）

### 説明
グリッド座標に基づいて入力テンソルからサンプリングします。Spatial Transformer Networksの核となる演算です。

### 動作原理

```mermaid
graph LR
    subgraph "入力"
        X["入力画像<br/>[N, C, H_in, W_in]"]
        G["グリッド座標<br/>[N, H_out, W_out, 2]"]
    end

    subgraph "GridSample"
        GS["bilinear/nearest<br/>補間"]
    end

    subgraph "出力"
        Y["変形画像<br/>[N, C, H_out, W_out]"]
    end

    X --> GS
    G --> GS
    GS --> Y

    style GS fill:#667eea,color:#fff
```

### 属性

| 属性名 | 型 | 説明 |
|--------|-----|------|
| mode | string | 補間モード ('bilinear', 'nearest', 'bicubic') |
| padding_mode | string | 範囲外処理 ('zeros', 'border', 'reflection') |
| align_corners | int | コーナーアラインメント |

### 主な用途
- **Spatial Transformer Networks**
- **画像変形・歪み補正**
- **オプティカルフロー適用**

---

## RoiAlign（ROIアライン）

### 説明
Region of Interest (ROI) から固定サイズの特徴を抽出します。バイリニア補間により量子化誤差を軽減します。

### 動作原理

```mermaid
graph TD
    subgraph "RoiAlign"
        F["特徴マップ"]
        R["ROI座標"]
        B["ビン分割"]
        S["サンプリング点"]
        P["平均/最大プーリング"]
        O["固定サイズ出力"]
    end

    F --> B
    R --> B
    B --> S --> P --> O

    style S fill:#667eea,color:#fff
```

### 主な用途
- **Mask R-CNN**
- **物体検出**
- **インスタンスセグメンテーション**

---

## Attention（アテンション）

### 説明
マルチヘッドアテンション機構を実装します。Transformerの核となる演算です。

### 動作原理

```mermaid
graph TD
    subgraph "Multi-Head Attention"
        Q["Query"]
        K["Key"]
        V["Value"]

        S["Scores = Q @ K^T / √d"]
        M["Mask (optional)"]
        SM["Softmax"]
        O["Output = SM @ V"]
    end

    Q --> S
    K --> S
    S --> SM
    M --> SM
    SM --> O
    V --> O

    style S fill:#667eea,color:#fff
    style SM fill:#f5576c,color:#fff
```

### 数式
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 主な用途
- **Transformer**: BERT, GPT等
- **自己注意機構**
- **クロスアテンション**

---

## RotaryEmbedding（RoPE）

### 説明
回転位置埋め込みを適用します。LLaMA、GPT-NeoXなどの大規模言語モデルで使用されます。

### 動作原理

```mermaid
graph LR
    subgraph "RoPE"
        X["入力ベクトル"]
        P["位置インデックス"]
        R["回転行列適用"]
        Y["位置埋め込み後"]
    end

    X --> R
    P --> R
    R --> Y

    style R fill:#667eea,color:#fff
```

### 数式
$$\text{RoPE}(x, pos) = \begin{bmatrix} x_0 \cos(\theta_0) - x_1 \sin(\theta_0) \\ x_0 \sin(\theta_0) + x_1 \cos(\theta_0) \\ \vdots \end{bmatrix}$$

### 特徴
- **相対位置を内積で表現**
- **長いコンテキストへの外挿が可能**
- **計算効率が高い**

### 主な用途
- **LLaMA, GPT-NeoX, PaLM**
- **長文処理**
- **言語モデル**

---

## ビット演算

### BitShift（ビットシフト）

```mermaid
graph LR
    subgraph "ビットシフト"
        A["X: 8 (0b1000)"]
        B["LEFT shift 2"]
        C["32 (0b100000)"]
    end

    A --> B --> C

    style B fill:#667eea,color:#fff
```

### Bitwise演算

| 演算子 | 数式 | 説明 |
|--------|------|------|
| BitwiseAnd | X & Y | 論理AND |
| BitwiseOr | X \| Y | 論理OR |
| BitwiseXor | X ^ Y | 排他的OR |
| BitwiseNot | ~X | 論理NOT |

### 主な用途
- **フラグ操作**
- **マスク処理**
- **ハッシュ計算**

---

## 特殊演算

### ReverseSequence（シーケンス反転）

```mermaid
graph LR
    subgraph "入力"
        X["[[1,2,3,4],<br/>[5,6,7,8]]"]
        L["lens: [3, 4]"]
    end

    subgraph "ReverseSequence"
        R[ReverseSequence]
    end

    subgraph "出力"
        Y["[[3,2,1,4],<br/>[8,7,6,5]]"]
    end

    X --> R
    L --> R
    R --> Y

    style R fill:#667eea,color:#fff
```

### Unique（一意要素）

テンソル内の一意な要素を返します。

### 出力
- **Y**: 一意な要素
- **indices**: 元のインデックス（オプション）
- **inverse_indices**: 逆インデックス（オプション）
- **counts**: 各要素のカウント（オプション）
