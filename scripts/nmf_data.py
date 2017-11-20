import numpy as np
import os
import argparse
from sklearn.decomposition import NMF

if __name__ == '__main__':
    parser = argparse.ArgumentParser( description='NMF Feature Extractor')
    parser.add_argument( '--comps', type = int, default = 700,
                        help='Number of components for NMF')
    parser.add_argument( '--solver', type = str, default = 'cd',
                        help='NMF Solver')
    args = parser.parse_args()

    ddir = '../data'
    data = np.array([])
    for n in range(1, 5):
        dir = '{0}/dataset_{1}'.format(ddir,n)
        for file in os.listdir(dir):
            D = np.load('{0}/{1}'.format(dir,file))
            D = D.reshape(D.shape[0], -1)
            if data.shape[0] == 0:
                data = np.copy(D)
            else:
                data = np.concatenate((data, D), axis = 0)
    print('Data Shape: ', data.shape)

    nmf = NMF(n_components=args.comps, solver=args.solver)
    transformed_data = nmf.fit_transform(data)
    print('Reconstruction Loss: ', nmf.reconstruction_err_)

    np.save('{0}/nmf_features.npy'.format(ddir),arr=nmf.components_)
    np.save('{0}/nmf_weights.npy'.format(ddir),arr=transformed_data)

