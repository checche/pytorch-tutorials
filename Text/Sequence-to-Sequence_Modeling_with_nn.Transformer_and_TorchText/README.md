# Sequence-to-Sequence Modeling with nn.Transformer and TorchText
nn.Transformerを使ったsequence-t-sequenceモデル学習

Transformer
- RNN系より並列化がしやすい
- sequence-to-sequenceのタスクの結果がよい

Attentionによって入出力間の大域的な依存関係を取れ得ることができる。

トークン: 単語  
シーケンス: 単語の並び

## モデル定義
言語モデルタスク（language modeling task）
- 特定の単語またはそのシーケンスが与えられた単語シーケンスのあとに続く確率を当てるタスク
- 前方のトークンのみに着目するために、入力シーケンスとともに、アテンションマスクが必要。(後方の内容は知る由もないので)
- マスクによって後方のトークンを未知のトークンとして扱う
- 出力は最終的に全結合層に送られ、log-softmax関数で処理される

nn.TransformerEncoder
- 複数のnn.TransformerEncoderLayerの層で構成されている。

Positionalencodingモジュール
- シーケンス内のトークンの位置に関する情報をモデルに与える
- 埋め込み層と同じ次元
- 位置エンコーディングの出力と埋め込み層の出力を足し算する事ができる。
- 今回はSinとCosで位置エンコーディングする

## データ読み込みとバッチ処理
vacabオブジェクトはトークンをテンソル形式の数値に変換する
`batchfy()`でシーケンス(トークンが左から順に横に並んだ形)をバッチ処理しやすく整形する。
バッチは独立したものとして扱われる。(ex. FGは連続しているがその関係は考慮できない。
```
[A B C D ... X Y Z]
=> [[A G M S]
    [B H N T]
    [C I O U]
    [D J P V]
    [E K Q W]
    [F L R X]]
```

### 入力シーケンスとTargetシーケンスを生成するための関数
`get_batch()`が入力とTargetを生成  
bpttはチャンクデータの長さ  
i番目の入力シーケンス`data[i]`のTargetシーケンスは`data[i+1]`である
