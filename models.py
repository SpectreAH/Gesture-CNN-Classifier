"""
Model architectures

    1. LinearRegression  — flattens + Linear, trained with MSE on one-hot 
    2. LogisticRegression — flattens + Linear, trained with Cross-Entropy
    3. MLP                — flattens + 2 hidden layers + Linear
    4. SmallCNN           — 2-block CNN 
    5. HandGestureCNN     — 4-block custom CNN 
"""
import torch.nn as nn


class LogisticRegression(nn.Module):
    """
    logistic regression: flattens so only has a single Linear layer.
    
    """

    def __init__(self, num_classes=10, in_channels=1, image_size=128):
        super().__init__()
        in_dim = in_channels * image_size * image_size
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_dim, num_classes) 

    def forward(self, x):
        return self.linear(self.flatten(x))
                                            

class LinearRegression(nn.Module):
    """
    Linear regression for classification: same architecture to
    LogisticRegression, but trained with MSE loss against one-hot targets. (One hot target means that in a category classification problem,
      the target is represented as a vector where the index corresponding to the correct class is set to 1, 
      and all other indices are set to 0. 
      For example, if there are 3 classes and the correct class is class 2, the one-hot target would be [0, 1, 0].)
    At inference, the predicted class is the index of the maximum value in the output.

    If should be a deliberately weak baseline if it not there is problem here
    """

    def __init__(self, num_classes=10, in_channels=1, image_size=128):
        super().__init__()
        in_dim = in_channels * image_size * image_size
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.linear(self.flatten(x))


class MLP(nn.Module):
    """
    Multi-Layer Perceptron: flatten -> [Linear -> ReLU -> Dropout] x N -> Linear.
    """

    def __init__(self, num_classes=10, in_channels=1, image_size=128,
                 hidden_dims=(256, 128), dropout=0.5):
        super().__init__()
        in_dim = in_channels * image_size * image_size

        layers = [nn.Flatten()]
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConvBlock(nn.Module):
    """Conv -> (BN) -> ReLU -> MaxPool. ()`use_bn` enables the BatchNorm ablation.)"""

    def __init__(self, in_ch, out_ch, use_bn=True, pool=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SmallCNN(nn.Module):
    """Shallow baseline: 2 conv blocks. Used to demonstrate depth matters."""

    def __init__(self, num_classes=10, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 16),
            ConvBlock(16, 32),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class HandGestureCNN(nn.Module):
    """
    Main custom CNN. With image_size=128 and in_channels=1:
    """

    def __init__(self, num_classes=10, in_channels=1, use_bn=True, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, use_bn=use_bn),
            ConvBlock(32, 64, use_bn=use_bn),
            ConvBlock(64, 128, use_bn=use_bn),
            ConvBlock(128, 256, use_bn=use_bn),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


REGRESSION_MODELS = {"linear"}  


def build_model(name, num_classes=10, image_size=128, **kwargs):
    """Factory: name in {linear, logistic, mlp, small, custom}."""
    name = name.lower()
    if name == "linear":
        return LinearRegression(num_classes=num_classes, image_size=image_size)
    if name == "logistic":
        return LogisticRegression(num_classes=num_classes, image_size=image_size)
    if name == "mlp":
        return MLP(num_classes=num_classes, image_size=image_size, **kwargs)
    if name == "small":
        return SmallCNN(num_classes=num_classes)
    if name == "custom":
        return HandGestureCNN(num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown model name: {name!r}")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
