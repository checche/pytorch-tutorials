import torch
import torch.nn as nn
import torchtext
from torchtext.data.utils import get_tokenizer

from src import datasets, models, trainers

if __name__ == '__main__':
    # 前処理とかやってくれるやつ
    TEXT = torchtext.data.Field(tokenize=get_tokenizer('basic_english'),
                                init_token='<sos>',
                                eos_token='<eos>',
                                lower=True)
    train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
    TEXT.build_vocab(train_txt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 20
    eval_batch_size = 10
    train_data = datasets.batchify(train_txt, batch_size, TEXT)
    val_data = datasets.batchify(val_txt, eval_batch_size, TEXT)
    test_data = datasets.batchify(test_txt, eval_batch_size, TEXT)

    print(train_data)
    # parameters for train
    ntokens = len(TEXT.vocab.stoi)  # the size of vacabulary
    emsize = 200  # embedding dimension
    nhid = 200  # nn.transformerEncoder内のfeedforward networkの次元数
    nlayers = 2
    nhead = 2  # multiheadattention modelのヘッドの数
    dropout = 0.2
    model = models.TransformerModel(
        ntokens,
        emsize,
        nhead,
        nhid,
        nlayers,
        dropout
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    lr = 5.0
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    trainer = trainers.Trainer(
        TEXT,
        train_data,
        val_data,
        test_data,
        35,
        model,
        criterion,
        optimizer,
        scheduler
    )

    trainer.fit()
