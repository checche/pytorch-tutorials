import math
import time

import torch

from src import datasets


class Trainer:
    def __init__(
        self,
        TEXT,
        train_data,
        val_data,
        test_data,
        bptt,
        model,
        criterion,
        optimizer,
        scheduler,
    ):
        self.TEXT = TEXT
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.bptt = bptt
        self.model = model
        self.best_model = None
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, epoch, model):
        model.train()
        total_loss = 0.
        start_time = time.time()
        ntokens = len(self.TEXT.vocab.stoi)
        for batch, i in enumerate(range(0, self.train_data.size(0) - 1, self.bptt)):
            data, targets = datasets.get_batch(self.train_data, i)
            self.optimizer.zero_grad()
            output = model(data)
            loss = self.criterion(output.view(-1, ntokens), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 0.5)  # 大きすぎる勾配をスケーリングする
            self.optimizer.step()

            total_loss += loss.item()
            log_interval = 200
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print(
                    '| epoch {:3d} | {:5d}/{:5d} batches | '
                    'lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, batch,
                        len(self.train_data) // self.bptt,
                        self.scheduler.get_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss))
                )
                total_loss = 0
                start_time = time.time()

    def evaluate(self, eval_model, data_source, TEXT, criterion):
        eval_model.eval()
        total_loss = 0.
        ntokens = len(TEXT.vacab.stoi)
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, self.bptt):
                data, targets = datasets.get_batch(data_source, i)
                output = eval_model(data)
                output_flat = output.view(-1, ntokens)
                total_loss += len(data) * \
                    criterion(output_flat, targets).item()
        return total_loss / (len(data_source) - 1)

    def fit(self):
        best_val_loss = float('inf')
        epochs = 3

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            self.train(epoch, self.model)
            val_loss = self.evaluate(self.model, self.val_data)
            print('-' * 89)
            print(
                '| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} |'
                'valid ppl {:8.2f}'.format(
                    epoch, (time.time() - epoch_start_time),
                    val_loss, math.exp(val_loss)
                )
            )
            print('-' * 89)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_model = self.model

            self.scheduler.step()

    def test(self):
        """ベストモデルでテストデータを評価"""
        test_loss = self.evaluate(self.best_model, self.test_data)
        print('=' * 89)
        print(
            '| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
                test_loss, math.exp(test_loss))
        )
        print('=' * 89)
