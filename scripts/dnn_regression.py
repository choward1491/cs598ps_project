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
import random
import visdom
import pdb

# Install visdom
# Make sure to start the server BEFORE running the file
# Start visdom server by running the following command: python -m visdom.server
# Use browser to access plots

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
    def __init__(self, file, train_val_split=0.8, transform=None):
        '''
        file : (string) pickle file name
        '''
        f = open(file, 'rb')
        patient_data = pickle.load(f)
        random.shuffle(patient_data)
        self.transform = transform
        num_patients = len(patient_data)
        sample_end = int(train_val_split * num_patients)
        self.train = patient_data[:sample_end]
        self.val = patient_data[sample_end:]

    def __len__(self):
        return len(self.train)

    def __getitem__(self, i):
        patient = self.train[i]
        label = patient.isControl
        features = patient.weights

        if self.transform:
            features = self.transform(features)

        return {'label': torch.ByteTensor([int(label)]),
                'features' : torch.FloatTensor(features)}

    def lenval(self):
        return len(self.val)

    def getval(self, i):
        patient = self.val[i]
        label = patient.isControl
        features = patient.weights

        if self.transform:
            features = self.transform(features)

        return {'label': torch.ByteTensor([int(label)]),
                'features' : torch.FloatTensor(features)}

class DNetwork(nn.Module):
    def __init__(self, input_size = (68, 20), activation = F.relu):
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
        fc_size = (5 * input_size[0] * input_size[1]) // 16
        self.fc1 = nn.Linear(fc_size, 100)
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
    parser.add_argument('--croplength', '-cl', type=int, default=68)
    parser.add_argument('--batchsize', '-bs', type=int, default=10)
    parser.add_argument('--transform', '-t', type=str, default='nmf')
    args = parser.parse_args()

    vis = visdom.Visdom(port=8097)
    datafile = '../data/nmf.pkl'
    if args.transform == 'nmf':
        datafile = '../data/nmf.pkl'
    elif args.transform == 'pca':
        datafile = '../data/pca.pkl'
    else:
        raise ValueError('Transform argument not recognized')

    # Get dataset
    dataset = PatientDataset(datafile, transform = TimeCrop(args.croplength))
    dataloader = DataLoader(dataset, batch_size = args.batchsize, shuffle = True)
    print('Length of Dataset: ', len(dataset))
    patient_data = dataset[np.random.randint(0, len(dataset))]
    label = patient_data['label']
    features = patient_data['features']
    print('Random Patient Information: {}, {}'.format(label, features.shape))
    dnn = DNetwork(input_size = (features.shape[0], features.shape[1]))
    loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(dnn.parameters(), weight_decay = 1e-4)

    depression_prop = 0
    for data in dataset:
        depression_prop += data['label'].numpy()[0]
    depression_prop /= len(dataset)
    print('Proportion of Depressed: ', depression_prop)

    # Run and optimize dnn
    train_costs = []
    val_costs = []
    epochs = []
    for epoch in range(args.epochs):
        total_accuracy = 0
        total_cost = 0
        dnn.train()
        for batch_count, batch in enumerate(dataloader):
            x = Variable(batch['features'].unsqueeze(1))
            y = Variable(batch['label'])
            logits = dnn(x)
            cost = loss(logits, y.float())

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Compute metrics
            estimates = logits > 0
            total_accuracy += torch.mean((estimates == y).double()).data.numpy()[0]
            total_cost += cost.data.numpy()[0]

        val_accuracy = 0
        val_cost = 0
        dnn.eval()
        for i in range(dataset.lenval()):
            x = Variable(dataset.getval(i)['features'].unsqueeze(0).unsqueeze(0))
            y = Variable(dataset.getval(i)['label']).unsqueeze(0)
            logits = dnn(x)
            cost = loss(logits, y.float())

            estimates = logits > 0
            val_accuracy += (estimates == y).data.numpy()[0][0]
            val_cost += cost.data.numpy()[0]

        print('Train Accuracy: ', total_accuracy / (batch_count + 1))
        print('Validation Accuracy: ', val_accuracy / (dataset.lenval() + 1))
        train_costs.append(total_cost / (batch_count + 1))
        val_costs.append(val_cost / (dataset.lenval() + 1))
        epochs.append(epoch)
        traces = [
            dict(x=epochs, y=train_costs, name='Training Loss', line={'width':1},
            mode='lines', type='scatter'),
            dict(x=epochs, y=val_costs, name='Validation Loss', line={'width':1},
            mode='lines', type='scatter'),
        ]
        layout = dict(
            showlegend=True,
            legend=dict( orientation='h', y=1.1, bgcolor='rgba(0,0,0,0)'),
            margin=dict( r=30, b=40, l=50, t=50),
            font=dict( family='Bell Gothic Std'),
            xaxis=dict( autorange=True, title='Epochs'),
            yaxis=dict( autorange=True, title='Loss'),
            title='Losses',
        )
        vis._send( dict( data=traces, layout=layout, win='loss'))
    vis.close()

if __name__ == '__main__':
    main()
