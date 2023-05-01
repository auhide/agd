"""
Trying out my implementation of the Automatic Gradient Descent. It is implemented
in a similar fashion to the standard PyTorch optimizers.
"""
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from agd import AGD


# Controlling the randomness in PyTorch and NumPy.
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.backends.cudnn.benchmark = True
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)


class TestNet(nn.Module):
    """Simple MLP network used for the testing of AGD.

    Args:
        in_size (int): The input layer size.
        out_size (int): The output layer size.
    """

    def __init__(self, in_size, out_size):
        super().__init__()

        # Since AGD doesn't support biases, we exclude them.
        self.l1 = nn.Linear(in_size, 4 * in_size, bias=False)
        self.l2 = nn.Linear(4 * in_size, out_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.l2(self.l1(x))

        return out


class TestDataset(Dataset):

    def __init__(self, X: torch.Tensor, y: torch.LongTensor):
        self.X, self.y = X, y

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)


def train(
    model: nn.Module, 
    dataloader: DataLoader, 
    loss_func: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    epochs: int = 1, 
    device: str = "cpu"
):

    for epoch in range(epochs):
        progressbar = tqdm(dataloader)

        for X, y in progressbar:
            optimizer.zero_grad()
            
            X, y = X.to(device), y.to(device)
            # X.shape: (BATCH_SIZE, N_FEATURES)
            # y.shape: (BATCH_SIZE)
            y_pred = model(X)
            # y_pred.shape: (BATCH_SIZE, N_CLASSES)

            loss = loss_func(y_pred, y)
            loss.backward()

            optimizer.step()

            progressbar.set_description(f"Epoch: {epoch + 1}/{epochs}, Loss: {loss.item():.2f}")


# Creating a random classification dataset.
N_FEATURES, N_CLASSES = 40, 2
X, y = make_classification(
    n_samples=1000, 
    n_informative=30,
    n_features=N_FEATURES, 
    n_classes=N_CLASSES
)
X, y = torch.Tensor(X), torch.LongTensor(y)
# Splitting the dataset into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8)

train_dataset = TestDataset(X_train, y_train)
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)

valid_dataset = TestDataset(X_valid, y_valid)
valid_dataloader = DataLoader(
    dataset=valid_dataset,
    batch_size=64,
    shuffle=True
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = TestNet(in_size=N_FEATURES, out_size=N_CLASSES).to(DEVICE)
loss = nn.CrossEntropyLoss()
# Initializing the Automatic Gradient Descent optimizer.
optimizer = AGD(model.parameters())

# Starting a training session.
train(
    model=model,
    dataloader=train_dataloader,
    loss_func=loss,
    optimizer=optimizer,
    epochs=3,
    device=DEVICE
)


# Since the dataset is not complex, even after the first epoch the loss is less
# than 1. You can test the optimizer with more complex data.

# Expected training output of the current setup:
# Epoch: 1/3, Loss: 0.93: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:02<00:00,  6.48it/s]
# Epoch: 2/3, Loss: 0.69: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 416.06it/s]
# Epoch: 3/3, Loss: 0.41: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 243.49it/s]
