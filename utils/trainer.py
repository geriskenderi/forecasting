import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Set up CUDA

class Trainer:
    # A helper class to train PyTorch DL models
    def __init__(self, model, loss_fn, optimizer, scheduler = None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, train_seq, n_epochs):
        for i in range(n_epochs):
            for in_seq, out_seq in train_seq:

                # Fix this part here to adapt to all inputs and make a standard input shape
                curr_input = in_seq.view(-1, len(in_seq), in_seq.size(1)).to(device)
                curr_out = out_seq.view(-1, len(out_seq), out_seq.size(1)).to(device)

                self.optimizer.zero_grad()
                y_pred = self.model(x=curr_input, future=len(out_seq))
                curr_loss = self.loss_fn(y_pred, curr_out)
                curr_loss.backward()
                self.optimizer.step()

            if self.scheduler is not None:
                # adjust learning rate
                self.lr_scheduler.step()

            print(f'Epoch: {i+1:3}, LR: {self.optimizer.param_groups[0]["lr"]}, Loss: {curr_loss.item():10.5f}')

        print(f'Loss after training: {curr_loss.item():10.5f}')