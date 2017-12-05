# -*- coding: utf-8 -*-
"""
@author: choward
@author: schan
"""

import pickle
import numpy as np

class Weights():
    def __init__(self, scan):
        self.isControl  = scan.isControl
        self.personId   = scan.personId
        self.runNo      = scan.runNo
        self.weights    = 0
    def setWeights(self, weights):
        self.weights=weights

def main():

    # load the main dataset and break into the important pieces
    mat= np.zeros(shape=(0, 16*16*10))
    weightsArr=[]
    arr = []
    try:
        f = open('../data/coarse.pkl', 'rb')
    except IOError:
        print("Could not open the file!")

    while True:
        try:
            scan = pickle.load(f)
        except:
            print("Scan failed!")
            break
        d=scan.data
        arr.append(d.shape[0])
        weightsArr.append(Weights(scan))
        mat = np.append(mat, np.reshape(d, (d.shape[0], d.shape[1]*d.shape[2]*d.shape[3])), axis=0)
        if len(arr)%20 == 0:
            print(len(arr))

    # save the
    np.save("../data/raw_features.npy", mat)

    # get the labels for each quantity
    num_weights     = len(weightsArr)
    control_lbls    = np.zeros((mat.shape[0],),dtype=int)
    id_lbls         = np.zeros((mat.shape[0],),dtype=int)

    # loop through the data points and set the labels
    sidx = 0
    for k in range(0,num_weights):
        control_lbls[sidx:(sidx+arr[k])] = weightsArr[k].isControl
        id_lbls[sidx:(sidx+arr[k])]      = weightsArr[k].personId
        sidx += arr[k]

    # save the labels
    np.save("../data/control_lbls.npy", control_lbls)
    np.save("../data/id_lbls.npy", id_lbls)

if __name__ == '__main__':
    class Scan():
        def __init__(self, isControl, personId, runNo, data):
            self.isControl = isControl
            self.personId = personId
            self.runNo = runNo
            self.data = data
    main()
