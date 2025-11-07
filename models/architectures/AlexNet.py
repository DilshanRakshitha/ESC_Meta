import torch
from torch import nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(
            self,
            input_size,
            num_classes = 26
        ):
        super(AlexNet, self).__init__()

        self.Conv2d_1 = nn.Conv2d(input_size, 128, kernel_size=11, stride=4)
        self.BatchNorm_1 = nn.BatchNorm2d(128)
        self.MaxPool_1 = nn.MaxPool2d(kernel_size=2)

        self.Conv2d_2 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2)
        self.BatchNorm_2 = nn.BatchNorm2d(256)
        self.MaxPool_2 = nn.MaxPool2d(kernel_size=3)

        self.Conv2d_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.BatchNorm_3 = nn.BatchNorm2d(256)

        self.Conv2d_4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.BatchNorm_4 = nn.BatchNorm2d(256)

        self.Conv2d_5 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.BatchNorm_5 = nn.BatchNorm2d(256)

        self.MaxPool_3 = nn.MaxPool2d(kernel_size=2)

        self.Flatten = nn.Flatten()
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.Dense_1 = nn.Linear(256 * 6 * 6, 1024)
        self.Dropout_1 = nn.Dropout(0.5)

        self.Dense_2 = nn.Linear(1024, 1024)
        self.Dropout_2 = nn.Dropout(0.5)

        self.Dense_3 = nn.Linear(1024, num_classes)


    def forward(self, x):
        x = F.relu(self.Conv2d_1(x))
        x = self.BatchNorm_1(x)
        x = self.MaxPool_1(x)

        x = F.relu(self.Conv2d_2(x))
        x = self.BatchNorm_2(x)
        x = self.MaxPool_2(x)

        x = F.relu(self.Conv2d_3(x))
        x = self.BatchNorm_3(x)

        x = F.relu(self.Conv2d_4(x))
        x = self.BatchNorm_4(x)

        x = F.relu(self.Conv2d_5(x))
        x = self.BatchNorm_5(x)

        x = self.MaxPool_3(x)
        
        x = self.adaptive_pool(x)

        x = self.Flatten(x)

        x = F.relu(self.Dense_1(x))
        x = self.Dropout_1(x)

        x = F.relu(self.Dense_2(x))
        x = self.Dropout_2(x)

        x = self.Dense_3(x)  # final output (logits)
        return x