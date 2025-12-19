from torch import nn
from torchinfo import summary
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 6, 1),   nn.BatchNorm2d(6),  nn.ReLU(),
            nn.Conv2d(6, 12, 1),  nn.BatchNorm2d(12), nn.ReLU(),
            nn.Conv2d(12, 24, 1), nn.BatchNorm2d(24), nn.ReLU(),
            nn.Conv2d(24, 48, 1), nn.BatchNorm2d(48), nn.ReLU(),
            nn.Conv2d(48, 64, 1), nn.BatchNorm2d(64), nn.ReLU(),
        )

        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.do1 = nn.Dropout2d(p=0.3)

        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)

        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.stem(x)

        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.do1(x)
        x = self.pool(F.relu(self.conv3(x)))

        x = F.relu(self.conv4(x))
        x = self.bn2(x)
        x = self.do1(x)
        x = F.relu(self.conv5(x))
        x = self.pool(F.relu(self.conv6(x)))

        x = F.relu(self.conv7(x))
        x = self.bn3(x)
        x = self.do1(x)
        x = F.relu(self.conv8(x))
        x = self.pool(F.relu(self.conv9(x)))

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        return x

if __name__=='__main__':
    model = CNN()
    summary(model, input_size=(64,3,32,32), col_names=["output_size", "num_params"], verbose=2)