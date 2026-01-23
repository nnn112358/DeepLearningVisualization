# ONNX演算子 - 畳み込み演算 (Convolution Operations)

畳み込み演算は画像認識やコンピュータビジョンの基盤となる演算です。局所的な特徴を抽出し、空間的な階層構造を学習することができます。

## 概要図

```mermaid
graph TB
    subgraph "畳み込み演算の種類"
        A[畳み込み] --> B[Conv<br/>標準畳み込み]
        A --> C[ConvTranspose<br/>転置畳み込み]

        B --> B1[1D Conv]
        B --> B2[2D Conv]
        B --> B3[3D Conv]

        B --> D[特殊な畳み込み]
        D --> D1[Depthwise Conv]
        D --> D2[Pointwise Conv]
        D --> D3[Dilated Conv]
    end
```

---

## Conv（畳み込み）

### 説明
畳み込み演算を行います。カーネル（フィルタ）を入力テンソル上でスライドさせながら、要素ごとの積和を計算して特徴マップを生成します。画像認識の基本となる演算で、局所的な特徴を抽出します。

1980年代のNeocognitronに起源を持ち、1998年のLeNet、2012年のAlexNetを経て、現代のディープラーニングの中核技術となりました。

### 畳み込みの動作原理

```mermaid
graph LR
    subgraph "入力画像"
        I["5×5<br/>入力"]
    end

    subgraph "カーネル"
        K["3×3<br/>フィルタ"]
    end

    subgraph "出力"
        O["3×3<br/>特徴マップ"]
    end

    I --> |"スライド＆積和"| O
    K --> |"重み"| O

    style K fill:#667eea,color:#fff
```

### 2D畳み込みの計算

```mermaid
graph TD
    subgraph "計算過程"
        A["入力領域<br/>[1,2,3]<br/>[4,5,6]<br/>[7,8,9]"]
        B["カーネル<br/>[a,b,c]<br/>[d,e,f]<br/>[g,h,i]"]
        C["出力値 =<br/>1a+2b+3c+<br/>4d+5e+6f+<br/>7g+8h+9i"]
    end

    A --> |"要素ごとの積"| C
    B --> |"重み"| C
```

### 数式
$$Y[n,c_{out},h,w] = \sum_{c_{in}} \sum_{k_h} \sum_{k_w} X[n,c_{in},h+k_h,w+k_w] \times W[c_{out},c_{in},k_h,k_w] + B[c_{out}]$$

### 入出力仕様

| 項目 | 名前 | 形状 | 説明 |
|------|------|------|------|
| 入力 | X | [N, C, H, W] | 入力テンソル (NCHW形式) |
| 入力 | W | [M, C/g, kH, kW] | 重みテンソル |
| 入力 | B | [M] | バイアス（オプション） |
| 出力 | Y | [N, M, oH, oW] | 出力テンソル |

### 属性

| 属性名 | 型 | デフォルト | 説明 |
|--------|-----|----------|------|
| auto_pad | string | "NOTSET" | パディング方式 |
| dilations | ints | [1, 1] | 膨張率 |
| group | int | 1 | グループ畳み込みのグループ数 |
| kernel_shape | ints | - | カーネルサイズ |
| pads | ints | [0,0,0,0] | パディング [top, left, bottom, right] |
| strides | ints | [1, 1] | ストライド |

### 出力サイズの計算

```
output_height = (input_height + pad_top + pad_bottom - kernel_height) / stride_h + 1
output_width  = (input_width + pad_left + pad_right - kernel_width) / stride_w + 1
```

### パディングモード

```mermaid
graph TD
    subgraph "auto_pad オプション"
        A["NOTSET"] --> A1["手動でpads指定"]
        B["SAME_UPPER"] --> B1["出力 = 入力サイズ<br/>余りは上/左に"]
        C["SAME_LOWER"] --> C1["出力 = 入力サイズ<br/>余りは下/右に"]
        D["VALID"] --> D1["パディングなし"]
    end
```

### ストライドとパディングの効果

```mermaid
graph LR
    subgraph "stride=1, pad=0"
        A1["5×5入力"] --> B1["3×3出力"]
    end

    subgraph "stride=2, pad=0"
        A2["5×5入力"] --> B2["2×2出力"]
    end

    subgraph "stride=1, pad=1"
        A3["5×5入力"] --> B3["5×5出力"]
    end
```

### Dilated Convolution（膨張畳み込み）

```mermaid
graph LR
    subgraph "dilation=1（通常）"
        D1["[x x x]<br/>[x x x]<br/>[x x x]"]
    end

    subgraph "dilation=2"
        D2["[x . x . x]<br/>[. . . . .]<br/>[x . x . x]<br/>[. . . . .]<br/>[x . x . x]"]
    end
```

膨張畳み込みは、カーネルの要素間に間隔を設けることで、パラメータを増やさずに受容野を拡大します。セマンティックセグメンテーションで有効です。

### グループ畳み込み

```mermaid
graph TD
    subgraph "通常の畳み込み (group=1)"
        A["入力: 6ch"] --> B["出力: 12ch"]
        B1["全チャネルが<br/>全チャネルに接続"]
    end

    subgraph "グループ畳み込み (group=2)"
        C["入力: 6ch"] --> D["グループ1: 3ch"]
        C --> E["グループ2: 3ch"]
        D --> F["出力1: 6ch"]
        E --> G["出力2: 6ch"]
    end

    subgraph "Depthwise (group=C)"
        H["入力: 6ch"] --> I["各チャネル独立"]
        I --> J["出力: 6ch"]
    end
```

### 使用例

```python
# 典型的なCNN層
入力: [1, 3, 224, 224]    # 1枚のRGB画像
カーネル: [64, 3, 7, 7]   # 64個の7×7フィルタ
ストライド: [2, 2]
パディング: [3, 3, 3, 3]
出力: [1, 64, 112, 112]   # 64チャネルの特徴マップ
```

### 畳み込みの種類

| 種類 | 説明 | 用途 |
|------|------|------|
| 1D Conv | 時系列データ | 音声、テキスト |
| 2D Conv | 画像データ | 画像認識（最も一般的） |
| 3D Conv | 動画、ボリューム | 動画分類、医療画像 |
| Depthwise | チャネルごとに独立 | MobileNet（軽量化） |
| Pointwise | 1×1カーネル | チャネル数変換 |

---

## ConvTranspose（転置畳み込み）

### 説明
畳み込みの転置演算で、主にアップサンプリングに使用されます。「逆畳み込み」や「デコンボリューション」とも呼ばれますが、厳密には畳み込みの逆演算ではなく、畳み込みの勾配計算に相当します。

入力より大きい出力を生成でき、エンコーダ-デコーダ構造のデコーダ部分や生成モデルで広く使用されます。

### 動作原理

```mermaid
graph LR
    subgraph "Conv (ダウンサンプリング)"
        A["4×4"] --> B["2×2"]
    end

    subgraph "ConvTranspose (アップサンプリング)"
        C["2×2"] --> D["4×4"]
    end

    B -.->|"形状の逆"| C

    style B fill:#667eea,color:#fff
    style D fill:#f5576c,color:#fff
```

### 処理の詳細

```mermaid
graph TD
    subgraph "ConvTranspose の動作"
        A["入力 2×2"] --> B["ゼロ挿入<br/>(stride分)"]
        B --> C["パディング追加"]
        C --> D["通常の畳み込み"]
        D --> E["出力 4×4"]
    end
```

### 出力サイズの計算

```
output_size = (input_size - 1) × stride - 2 × padding + kernel_size + output_padding
```

### 入出力仕様

| 項目 | 名前 | 形状 | 説明 |
|------|------|------|------|
| 入力 | X | [N, C, H, W] | 入力テンソル |
| 入力 | W | [C, M/g, kH, kW] | 重みテンソル |
| 入力 | B | [M] | バイアス（オプション） |
| 出力 | Y | [N, M, oH, oW] | 出力テンソル |

### 属性

| 属性名 | 型 | デフォルト | 説明 |
|--------|-----|----------|------|
| auto_pad | string | "NOTSET" | パディング方式 |
| dilations | ints | [1, 1] | 膨張率 |
| group | int | 1 | グループ数 |
| kernel_shape | ints | - | カーネルサイズ |
| output_padding | ints | [0, 0] | 出力パディング |
| output_shape | ints | - | 出力形状（直接指定） |
| pads | ints | [0,0,0,0] | パディング |
| strides | ints | [1, 1] | ストライド |

### チェッカーボードアーティファクト

```mermaid
graph TD
    A["問題"] --> B["ストライドとカーネルサイズが<br/>割り切れない場合"]
    B --> C["格子状のパターンが出現"]

    D["解決策"] --> E["kernel_size を<br/>stride の倍数に"]
    D --> F["Resize + Conv で<br/>代替"]
```

### 使用例

```python
# セマンティックセグメンテーション（2倍アップサンプリング）
入力: [1, 64, 14, 14]
カーネル: [64, 32, 4, 4]
ストライド: [2, 2]
出力: [1, 32, 28, 28]
```

### 主な用途

```mermaid
graph TD
    A[ConvTranspose] --> B[セマンティックセグメンテーション<br/>U-Net, FCN]
    A --> C[画像生成<br/>GAN, VAE]
    A --> D[超解像<br/>SRCNN]
    A --> E[オートエンコーダ<br/>デコーダ部分]
```

---

## CNNアーキテクチャでの使用

```mermaid
graph LR
    subgraph "典型的なCNN"
        A["入力<br/>224×224"] --> B["Conv + Pool<br/>112×112"]
        B --> C["Conv + Pool<br/>56×56"]
        C --> D["Conv + Pool<br/>28×28"]
        D --> E["Conv + Pool<br/>14×14"]
        E --> F["Conv + Pool<br/>7×7"]
        F --> G["FC層"]
        G --> H["出力"]
    end
```

```mermaid
graph LR
    subgraph "U-Net (エンコーダ-デコーダ)"
        A["入力"] --> B["エンコーダ<br/>(Conv)"]
        B --> C["ボトルネック"]
        C --> D["デコーダ<br/>(ConvTranspose)"]
        D --> E["出力"]

        B -.->|"スキップ接続"| D
    end
```

---

## パフォーマンス考慮事項

| 要素 | 影響 | 最適化 |
|------|------|--------|
| カーネルサイズ | 大きいほど計算量増 | 3×3を多用 |
| チャネル数 | 計算量に二次的に影響 | ボトルネック構造 |
| ストライド | 大きいほど出力小 | ダウンサンプリングに活用 |
| グループ | パラメータ削減 | Depthwise Separable |
| Dilation | 受容野拡大 | セグメンテーション |
