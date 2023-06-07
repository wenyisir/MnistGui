from torch.utils.data import Dataset

class NumberDataset(Dataset):
    def __init__(self, img, transform=None):
        super(Dataset, self).__init__()
        self.transform = transform
        self.img = img

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        img = self.img
        img = img.reshape(1, img.shape[0], -1)
        return img


import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.nn as nn


# coding: utf-8
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__() # 父类构造方法
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input image size: [1, 32, 32]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),            # [1,32,32]->[64,32,32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),                # [64,32,32]->[64,16,16]

            nn.Conv2d(64, 128, 3, 1, 1),          #[64,16,16]->[128,16,16]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),                #[128,16,16]->[128,8,8]

            nn.Conv2d(128, 256, 3, 1, 1),         #[128,8,8]->[256,8,8]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),                #[256,8,8]->[256,4,4]
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 10]
        x = self.cnn_layers(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        return x


def number_recogntion(img):

    img = torch.tensor(img, dtype=torch.float32)
    test_set = NumberDataset(img)
    batch_size = 1
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNN().to(device)
    model.device = device
    model.load_state_dict(torch.load('cnn_para.mdl'))
    model.eval()

    predictions = []
    for batch in tqdm(test_loader):
        imgs = batch
        with torch.no_grad():
            logits = model(imgs.to(device))
        predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

    return predictions






