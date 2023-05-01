# Automatic Gradient Descent PyTorch Optimizer Implementation
Implementation of the Automatic Gradient Descent algorithm from [this](https://arxiv.org/pdf/2304.05187.pdf) paper. This is an optimizer for which no hyperparameters are needed. The learning rate is tuned as a hyperparameter itself.

**Please note that there is an original implementation by the creators of the algorithm.** I went through the paper and implemented the optimizer in the style of PyTorch's optimizers. I also tried to comment each step of the algorithm in `agd.py`. In `main.py` I tested out the optimizer with a sample classification dataset and a cross-entropy loss.

## Usage
`AGD` can be used exactly as all PyTorch optimizers, since it inherits `Optimizer` as well.
```python
from torch import nn

from agd import AGD


model = MyModel()
# You can also use MSE.
loss_func = nn.CrossEntropyLoss()
# No hyperparameters are needed.
optimizer = AGD(model.parameters())


def train(...):
    for x, y in batches:
        optimizer.zero_grad()

        y_pred = model(x)
        loss = loss_func(y_pred, y)

        loss.backward()

        # Make an optimization step.
        optimizer.step()
```