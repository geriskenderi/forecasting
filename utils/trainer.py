import numpy as np


class Trainer:
    # A helper class to train PyTorch DL models
    def __init__(self, model, train_loader, test_loader, loss_fn, optimizer, device, scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train(self, n_epochs):
        for i in range(n_epochs):
            self.model.train()
            for in_seq, out_seq in self.train_loader:
                input = in_seq.to(self.device)
                output = out_seq.to(self.device)

                self.optimizer.zero_grad()
                y_pred = self.model(x=input, future=output.shape[1])
                loss = self.loss_fn(y_pred, output)
                loss.backward()
                self.optimizer.step()

            if self.scheduler is not None:
                # adjust learning rate
                self.scheduler.step()

            print(f'Epoch: {i + 1:3}, LR: {self.optimizer.param_groups[0]["lr"]}, Loss: {loss.item():10.5f}')

            # Evaluate model on the testing set every 10 epochs
            if i % 5 == 0:
                self.evaluate()

        print(f'Loss on training set after final training epoch: {loss.item():10.5f}')

    def evaluate(self):
        self.model.eval()
        for in_seq, out_seq in self.test_loader:
            input = in_seq.to(self.device)
            output = out_seq.to(self.device)
            y_pred = self.model(x=input, future=output.shape[1])
            loss = self.loss_fn(y_pred, output)
        print('************** EVAL **************')
        print(f'Loss on testing set: {loss.item():10.5f}')
        print('**********************************')

    def forecast(self):
        self.model.eval()
        gt, forecasts = [], []
        for in_seq, out_seq in self.test_loader:
            input = in_seq.to(self.device)
            output = out_seq.to(self.device)
            horizon, out_dim = output.shape[1], output.shape[2]
            y_pred = self.model(x=input, future=output.shape[1])
            forecasts.append(y_pred.detach().cpu().numpy())
            gt.append(output.detach().cpu().numpy())

        np_forecasts = np.array(forecasts).reshape(-1, horizon, out_dim)
        np_gt = np.array(gt).reshape(-1, horizon, out_dim)

        return np.squeeze(np_gt), np.squeeze(np_forecasts)
