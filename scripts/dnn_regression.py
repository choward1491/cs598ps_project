import pickle
from torch.utils.data import Dataset, DataLoader
from feature_extraction import Weights
import numpy as np
import pdb

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
    dataset = PatientDataset('../data/nmf.pkl')
    print('Length of Dataset', len(dataset))
    patient_data = dataset[np.random.randint(0, len(dataset))]
    label = patient_data['label']
    features = patient_data['features']
    print('Random Patient Information: {}, {}'.format(label, features.shape))

if __name__ == '__main__':
    main()
