import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def batchify(data, batch_size, TEXT):
    data = TEXT.numericalize([data.examples[0].text])
    nbatch = data.size(0) // batch_size
    # 例えば[A B C ... X Y Z] => [A B C ...V W X]
    # 指定した範囲にテンソルを削る
    data = data.narrow(0, 0, nbatch * batch_size)
    # -1はよしなに次元の大きさを調節するやつ
    # contiguousはメモリ上での値の連続性を復活させる関数
    data = data.view(batch_size, -1).t().contiguous()
    return data.to(device)


bptt = 35


def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target
