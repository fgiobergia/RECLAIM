import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
from torchvision import transforms
import os
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import do_runs

class ImageDataset:
    def __init__(self, classes):
        """
        classes: list of classes, using position for the numerical label
        (e.g. ['a','b'] will assign class 0 to 'a', class 1 to 'b')
        """
        classes_dirs = {
            pos: [ path
                  for paths in [
                      glob(os.path.join("../train", class_name, "*.JPEG"))
                      for class_name in class_names
                  ]
                  for path in paths
                 ]
            for pos, class_names in enumerate(classes)
        }
        
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        X = []
        y = []
        for class_label, class_path in classes_dirs.items():
            with tqdm(class_path) as bar:
                for fname in bar:
                    with Image.open(fname) as im:
                        im = im.convert("RGB")
                        X.append(preprocess(im))
                        y.append(class_label)
        
        self.X = torch.stack(X)
        self.y = torch.tensor(y)
        # No shuffling is done here. Expecting it to be performed by DataLoader!
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]

class ResNetBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.resnet_fe = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(2048, 1)
        
        for layer in self.resnet_fe.parameters():
            layer.requires_grad = False
    
    def forward(self, x):
        x = self.resnet_fe(x)
        x = x.squeeze()
        x = self.fc(x)
        return x.flatten()

    
def trainer(ds, random_state):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_size = int(len(ds) * .8)
    test_size = len(ds) - train_size
    ds_train, ds_test = random_split(ds, [train_size, test_size], generator=torch.Generator().manual_seed(random_state))

    dl_train = DataLoader(ds_train, batch_size=128)
    dl_test = DataLoader(ds_test, batch_size=len(ds_test))

    model = ResNetBinaryClassifier()
    model.to(device)
    opt = optim.Adam(model.parameters())
    loss_func = nn.BCEWithLogitsLoss()

    with tqdm(range(5)) as bar:
        for epoch in bar:
            cum_loss = 0
            for i, (X_batch, y_batch) in enumerate(dl_train):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                opt.zero_grad()
                y_pred = model(X_batch)

                loss = loss_func(y_pred, y_batch.float())

                loss.backward()
                opt.step()

                cum_loss += loss.item()
                bar.set_postfix(loss=cum_loss / (i+1))

    X_test, y_test = next(iter(dl_test))
    X_test = X_test.to(device)
    y_pred = torch.sigmoid(model(X_test)) > .5
    
    y_test = y_test.detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    C = len(ds_test)
    N_P = y_test.sum()
    metrics = {
        "A": accuracy_score(y_test, y_pred),
        "P": precision_score(y_test, y_pred),
        "R": recall_score(y_test, y_pred),
        "Fb": f1_score(y_test, y_pred)
    }
    return C, N_P, metrics
    
    
if __name__ == "__main__":
    # tabby cat: n02123045
    # egyptian cat: n02124075
    # n02110341: dalmatian, coach dog, carriage dog
    # n02097474: Tibetan terrier, chrysanthemum dog
    class0 = ["n02123045", "n02124075"]
    class1 = ["n02110341", "n02097474"]
    ds = ImageDataset([class0, class1])
    
    torch.manual_seed(42)
    
    do_runs(ds, trainer, 10)