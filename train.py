import torch
import torch.optim as optim

import dataset
import model

device = "cuda"

data = dataset.LentaDataBank()
data.load_dataset("./data/Lenta")

NUM_NEGATIVE_SAMPLES = 10
NUM_CONTEXT = 300000

dtset = dataset.Word2VecDataset(data, NUM_NEGATIVE_SAMPLES, NUM_CONTEXT)
dtset.generate_dataset()

nn_model = model.Word2Vec(data.num_tokens())
nn_model.type(torch.cuda.FloatTensor)
nn_model.to(device)

optimizer = optim.SGD(nn_model.parameters(), lr=0.05, weight_decay=0)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.003, patience=3)
train_loader = torch.utils.data.DataLoader(dtset, batch_size=64)

loss_history, acc_history = model.train(nn_model, dtset, train_loader, optimizer, num_epochs=10, scheduler=scheduler, scheduler_loss=True)

# сохраняет текущее состояние модели и тренировки
torch.save(nn_model.state_dict(), "checkpoint.pth")
