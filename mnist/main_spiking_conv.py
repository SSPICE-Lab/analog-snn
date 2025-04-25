import torch
import torchvision
from torch import nn

import torch_framework.spiking as spike

DATASET_PATH = "/data/shared/datasets/torch/"

def main(**kwargs):
    torch.manual_seed(42)

    # Hyperparameters
    TRAIN_VAL_SPLIT = 0.2
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 0.001

    # Neuron parameters
    TIMESTEPS = 100
    KVCO = 30e6 * 2 * torch.pi
    CENTER_FREQ = 119.46e6
    KPD = 1 / torch.pi
    THRESHOLD = -0.05
    ALPHA = 0.5
    TIMESTEP = 1 / CENTER_FREQ
    BIAS = False

    # Set device for PyTorch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: `{device}`\n")
    torch.backends.cudnn.benchmarks = True

    # Load MNIST train data
    # Pixels are in range [0, 1]
    full_dataset = torchvision.datasets.MNIST(DATASET_PATH, train=True, download=True,
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]
        )
    )

    # Split data into train and validation
    VAL_SIZE = int(TRAIN_VAL_SPLIT * len(full_dataset))
    TRAIN_SIZE = len(full_dataset) - VAL_SIZE
    train_data, val_data = torch.utils.data.random_split(full_dataset, [TRAIN_SIZE, VAL_SIZE])

    train_data.transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        torchvision.transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
        torchvision.transforms.RandomRotation(degrees=15),
        torchvision.transforms.ElasticTransform(alpha=1.0, sigma=0.07),
        torchvision.transforms.RandomPerspective(),
        torchvision.transforms.ToTensor()
    ])

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=1, pin_memory=torch.cuda.is_available())
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=1, pin_memory=torch.cuda.is_available())

    # Load test data
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(DATASET_PATH, train=False, download=True,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
            )
        ),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=1, pin_memory=torch.cuda.is_available())

    net = spike.network.Sequential((64, 28, 28, 1), KVCO, KPD, TIMESTEP, THRESHOLD, ALPHA, bias=BIAS)
    net.add(spike.layers.encoding.PoissonEncoding(TIMESTEPS, 1.0))
    net.add("conv2d", 32)
    net.add("maxpool2d")
    net.add("flatten")
    net.add("dropout", p=0.45)
    net.add("output", 10)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.9**(1/8))
    loss_fn = nn.CrossEntropyLoss(reduction="mean")

    save_path = f"saved_models/mnist_cnn_vco_{KVCO}_{TIMESTEP}_augment_32.pth"
    history = net.fit(
        train_loader,
        val_loader,
        EPOCHS,
        device,
        optimizer,
        loss_fn,
        lr_schedule,
        save_best_model=True,
        save_monitor="acc",
        save_filepath=save_path
    )

    net.load_state_dict(torch.load(save_path, map_location=device))

    loss, acc = net.evaluate(test_loader, device, loss_fn)
    print(f"Test data: loss = {loss:.3f}, accuracy = {100*acc:.2f}%")

    return loss, acc


if __name__ == "__main__":
    main()
