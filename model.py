import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

#WORDVEC_DIM_FOR_TEST = 10
WORDVEC_DIM = 256

class Word2Vec(nn.Module):
    def __init__(self, num_tokens):
        super(Word2Vec, self).__init__()
        self.input = nn.Linear(in_features=num_tokens, out_features=WORDVEC_DIM, bias=False).to(device)
        self.output = nn.Linear(in_features=WORDVEC_DIM, out_features=num_tokens, bias=False).to(device)

    def forward(self, input_index_batch, output_indices_batch):
        '''
        Arguments:
            input_index_batch - Tensor of ints, shape: (batch_size, ), indices of input words in the batch
            output_indices_batch - Tensor if ints, shape: (batch_size, num_negative_samples + 1),
                                   indices of the target words for every sample
                                
        Returns:
            predictions - Tensor of floats, shape: (batch_size, num_negative_samples + 1)
        '''
        
        predictions = []
        for i, input_index in enumerate(input_index_batch):
            output_indices = output_indices_batch[i]

            u = self.input.weight[:, input_index]
            v = self.output.weight[output_indices, :]

            prediction = torch.mv(v, u)
            predictions.append(prediction)

        return torch.stack(predictions)

def train(model, dataset, train_loader, optimizer, num_epochs, scheduler=None, scheduler_loss=False):
    loss = nn.BCEWithLogitsLoss().type(torch.FloatTensor)

    loss_history = []
    acc_history = []
    for epoch in range(num_epochs):
        model.train()

        dataset.generate_dataset()

        loss_accum = 0
        correct_samples = 0
        total_samples = 0

        for i_step, (x, indices, y_multi) in enumerate(train_loader):
            x_gpu = x.to(device)
            indices_gpu = indices.to(device)
            y_multi_gpu = y_multi.to(device)

            predictions = model(x_gpu, indices_gpu)
            loss_value = loss(predictions, y_multi_gpu)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            predictions_max = torch.argmax(predictions, 1)
            y_multi_max = torch.argmax(y_multi_gpu, 1)
            correct_samples += float(torch.sum(torch.eq(predictions_max, y_multi_max)))
            total_samples += float(y_multi.shape[0])

            loss_accum += float(loss_value)

        train_loss = loss_accum / i_step
        train_acc = float(correct_samples) / total_samples

        loss_history.append(train_loss)
        acc_history.append(train_acc)

        if scheduler is not None:
            if scheduler_loss:
                scheduler.step(train_loss)
            else:
                scheduler.step()

        print(f"Epoch #{epoch} - train loss: {train_loss}, accuracy: {train_acc}")

    return loss_history, acc_history
