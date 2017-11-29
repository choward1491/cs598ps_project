import pickle
from torch.utils.data import Dataset, DataLoader
from feature_extraction import Weights
import numpy as np
import pdb

class TimeCrop(object):
    '''
    Crop file along the time dimension so that all files are the same length.
    Time axis is assumed to be 0 (1st axis) of sample.
    '''
    def __init__(self, size, random = True):
        self.random = random
        self.size = size

    def __call__(self, sample):
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

def main():
    dataset = PatientDataset('../data/nmf.pkl', transform = TimeCrop(100))
    print('Length of Dataset', len(dataset))
    patient_data = dataset[np.random.randint(0, len(dataset))]
    label = patient_data['label']
    features = patient_data['features']
    print('Random Patient Information: {}, {}'.format(label, features.shape))

if __name__ == '__main__':
    main()
