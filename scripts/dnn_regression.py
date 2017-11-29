import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from feature_extraction import Weights
import argparse
import numpy as np
import pdb

class TimeCrop(object):
    '''
    Crop file along the time dimension so that all files are the same length.
    Time axis is assumed to be 0 (1st axis) of sample.
    '''
    def __init__(self, size = None, random = True):
        self.random = random
        self.size = size

    def __call__(self, sample):
        if not self.size:
            return sample

        assert sample.shape[0] >= self.size
        if self.random:
            start = np.random.randint(0, sample.shape[0] - self.size)
        else:
            start = 0
        return sample[start : start + self.size]

class PatientDataset(Dataset):
    def __init__(self, file, transform=None):
        '''
        file : (string) pickle file name
        '''
        f = open(file, 'rb')
        self.patient_data = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.patient_data)

    def __getitem__(self, i):
        patient = self.patient_data[i]
        label = patient.isControl
        features = patient.weights

        if self.transform:
            features = self.transform(features)

        return {'label': label, 'features' : features}

class DNetwork(nn.Module):
    def __init__(self, activation = F.relu):
        super(DNetwork, self).__init__()
        # Conv layers with batch norm
        self.activation = activation
        self.conv1 = nn.Conv2d(1, 5, 7, stride = 1, padding = 3)
        self.conv_bn1 = nn.BatchNorm2d(5)
        self.conv2 = nn.Conv2d(5, 5, 7, stride = 1, padding = 3)
        self.conv_bn2 = nn.BatchNorm2d(5)
        self.mp1 = nn.MaxPool2d(2)
        self.mp2 = nn.MaxPool2d(2)

        # Fully connected layers with batch norm
        self.fc1 = nn.Linear(400, 100)
        self.fc_bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        '''
        x : (torch Variable) expected size of (batch size, 1, time, features)
        '''
        batch_size = x.size()[0]
        # Conv layers
        h = self.activation(self.conv1(x))
        h = self.conv_bn1(h)
        h = self.activation(self.conv2(h))
        h = self.conv_bn2(h)
        h = self.mp1(h)
        h = self.mp2(h)

        # Fully connected layers
        h = h.view(batch_size, -1)
        h = self.fc1(h)
        h = self.fc_bn1(h)
        y = self.fc2(h)
        return y

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='DNN Classification')
    parser.add_argument('--epochs', '-e', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--croplength', '-cl', type=int, default=64)
    parser.add_argument('--batchsize', '-bs', type=int, default=10)
    args = parser.parse_args()

    # Get dataset
    dataset = PatientDataset('../data/nmf.pkl', transform = TimeCrop(args.croplength))
    dataloader = DataLoader(dataset, batch_size = args.batchsize, shuffle = True)
    print('Length of Dataset: ', len(dataset))
    patient_data = dataset[np.random.randint(0, len(dataset))]
    label = patient_data['label']
    features = patient_data['features']
    print('Random Patient Information: {}, {}'.format(label, features.shape))
    dnn = DNetwork().double()
    loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(dnn.parameters())

    for epoch in range(args.epochs):
        total_accuracy = 0
        total_loss = 0
        for batch_count, batch in enumerate(dataloader):
            x = Variable(batch['features'].unsqueeze(1))
            y = Variable(batch['label']).byte()
            logits = dnn(x).squeeze(1)
            loss_val = loss(logits, y.double())

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Compute metrics
            estimates = logits > 0
            total_accuracy += torch.mean((estimates == y).double()).data.numpy()[0]
            total_loss += loss_val.data.numpy()[0]
        print('Loss: ', total_loss / (batch_count + 1))
        print('Accuracy: ', total_accuracy / (batch_count + 1))

if __name__ == '__main__':
    main()
