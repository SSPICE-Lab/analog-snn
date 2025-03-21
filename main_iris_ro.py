import numpy as np
import torch
import tqdm
from sklearn.datasets import load_iris

import neurons


class SNN(torch.nn.Module):
    def __init__(self, kvco, ffree, vth, std_gain=None):
        super(SNN, self).__init__()
        self.fc1 = neurons.ro.RO_Dense(4, 16, kvco, ffree, vth, std_gain)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc3 = torch.nn.Linear(16, 3)

    def forward(self, x):
        spikes = self.fc1(x)
        spike_count = torch.sum(spikes, dim=1)
        x = self.fc3(spike_count)
        return x, torch.mean(spike_count)

if __name__ == '__main__':
    SEED = 1
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    TIMESTEPS = 200
    MIN_RATE = 0
    MAX_RATE = 0.1

    iris = load_iris()
    input_data = iris['data']
    target_data = iris['target']

    # Normalize data
    input_data = input_data - np.min(input_data)
    input_data = input_data / np.max(input_data)
    input_data = input_data * (MAX_RATE - MIN_RATE) + MIN_RATE

    idx = np.random.permutation(input_data.shape[0])
    n_train = int(input_data.shape[0] * 0.6)
    n_val = int(input_data.shape[0] * 0.2)

    train_data = input_data[idx[:n_train]]
    train_labels = target_data[idx[:n_train]]
    val_data = input_data[idx[n_train:n_train + n_val]]
    val_labels = target_data[idx[n_train:n_train + n_val]]
    test_data = input_data[idx[n_train + n_val:]]
    test_labels = target_data[idx[n_train + n_val:]]

    train_data = torch.Tensor(train_data).float()
    train_labels = torch.Tensor(train_labels).long()
    val_data = torch.Tensor(val_data).float()
    val_labels = torch.Tensor(val_labels).long()
    test_data = torch.Tensor(test_data).float()
    test_labels = torch.Tensor(test_labels).long()

    model = SNN(kvco=10e6, ffree=100e6, vth=-0.1, std_gain=60)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.995)

    pbar = tqdm.trange(1000)
    best_loss = float('inf')
    for epoch in pbar:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        input_spikes = torch.zeros(train_data.shape[0], TIMESTEPS, train_data.shape[-1]).float()
        rand_numbers = torch.rand(train_data.shape[0], TIMESTEPS, train_data.shape[-1])
        input_spikes[rand_numbers < train_data[:, np.newaxis, :]] = 0.2

        outputs, spike_count = model(input_spikes)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            input_spikes = torch.zeros(val_data.shape[0], TIMESTEPS, val_data.shape[-1]).float()
            rand_numbers = torch.rand(val_data.shape[0], TIMESTEPS, val_data.shape[-1])
            input_spikes[rand_numbers < val_data[:, np.newaxis, :]] = 0.2

            outputs, _ = model(input_spikes)
            loss = criterion(outputs, val_labels)
            correct = (torch.argmax(outputs, dim=1) == val_labels).type(torch.FloatTensor)
            accuracy = correct.mean().item()

            if loss <= best_loss:
                best_loss = loss
                torch.save(model.state_dict(), 'iris_model_vco.pth')

        pbar.set_description(f'Loss: {loss.item():.4f}')
        pbar.set_postfix(Accuracy=f'{accuracy * 100:.2f}%', Best_Loss=f'{best_loss:.4f}')

    model.load_state_dict(torch.load('iris_model_vco.pth', weights_only=True))

    model.eval()
    max_spike_rate = 0
    with torch.no_grad():
        correct = 0
        total = 0
        max_acc = 0
        for i in range(100):
            input_spikes = torch.zeros(test_data.shape[0], TIMESTEPS, test_data.shape[-1]).float()
            rand_numbers = torch.rand(test_data.shape[0], TIMESTEPS, test_data.shape[-1])
            input_spikes[rand_numbers < test_data[:, np.newaxis, :]] = 0.2

            outputs, spike_rate = model(input_spikes)
            correct_iter = (torch.argmax(outputs, dim=1) == test_labels).type(torch.FloatTensor).sum().item()
            if correct_iter > max_acc:
                max_spike_rate = spike_rate
                max_acc = correct_iter
            correct += correct_iter
            total += test_labels.shape[0]
        accuracy = correct / total
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Max Accuracy: {max_acc / test_labels.shape[0] * 100:.2f}%')
    print(f'Max Spike Rate: {max_spike_rate.item()*100/TIMESTEPS:.2f}%')
