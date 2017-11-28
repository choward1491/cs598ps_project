# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 09:28:41 2017

@author: schan
"""

import pickle
import numpy as np
from sklearn.decomposition import PCA, NMF

class Weights():
    def __init__(self, scan):
        self.isControl = scan.isControl
        self.personId=scan.personId
        self.runNo=scan.runNo
        self.weights=0
    def setWeights(self, weights):
        self.weights=weights

def main():
    mat= np.zeros(shape=(0, 16*16*10))
    weightsArr=[]
    arr = []
    f = open('C:\\Users\\schan\\Documents\\LargeFiles\\coarse.pkl', 'rb')
    while True:
        try:
            scan = pickle.load(f)
        except:
            break
        d=scan.data
        arr.append(d.shape[0])
        weightsArr.append(Weights(scan))
        mat = np.append(mat, np.reshape(d, (d.shape[0], d.shape[1]*d.shape[2]*d.shape[3])), axis=0)
        if len(arr)%20 == 0:
            print(len(arr))

    print(mat.shape)
    np.save("C:\\Users\\schan\\Documents\\LargeFiles\\mat.npy", mat)

    doPCA=False
    doNMF=True
    if doPCA:
        pca = PCA(n_components=700, whiten=True)
        pca.fit(mat)

        print("PCA Fitted")
        curr = 0
        for i,x in enumerate(arr):
            weights = pca.transform(mat[curr:(curr+x), :])
            curr+=x
            weightsArr[i].setWeights(weights)

        f = open('C:\\Users\\schan\\Documents\\LargeFiles\\pca.pkl', 'wb')
        pickle.dump(weightsArr,f,pickle.HIGHEST_PROTOCOL)
        pickle.dump(pca.components_, f, pickle.HIGHEST_PROTOCOL)
    if doNMF:
        nmf = NMF(n_components=20)
        nmf.fit(mat)

        print("NMF Fitted")
        curr = 0
        for i,x in enumerate(arr):
            weights = nmf.transform(mat[curr:(curr+x), :])
            curr+=x
            weightsArr[i].setWeights(weights)

        f = open('C:\\Users\\schan\\Documents\\LargeFiles\\nmf.pkl', 'wb')
        pickle.dump(weightsArr,f,pickle.HIGHEST_PROTOCOL)
        pickle.dump(nmf.components_, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
