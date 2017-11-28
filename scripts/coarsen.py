# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 10:20:25 2017

@author: manchan2
"""

import numpy as np
import nibabel as nib
import os
import pickle

class Scan():
    def __init__(self, isControl, personId, runNo, data):
        self.isControl=isControl
        self.personId=personId
        self.runNo=runNo
        self.data=data
        
output = open('C:\\Users\\schan\\Documents\\LargeFiles\\coarse.pkl', 'wb')
#import matplotlib.pyplot as plt
#from sklearn.decomposition import NMF 
import time
def runData(isControl, subj, run):
    if isControl:
        suffix="_bold.nii.gz"
        directory = "C:\\Users\\schan\\Documents\\LargeFiles\\ds171_R1.0.0\\sub-control"+str(subj).zfill(2)+"\\func"
        files = [x for x in os.listdir(directory) if x.endswith(suffix)]
        img=nib.load(directory+"\\"+sorted(files)[run-1])
        data=img.get_data()
        return(data)
    else:
        suffix="_bold.nii.gz"
        directory = "C:\\Users\\schan\\Documents\\LargeFiles\\ds171_R1.0.0_mdd\\ds171_R1.0.0\\sub-mdd"+str(subj).zfill(2)+"\\func"
        files = [x for x in os.listdir(directory) if x.endswith(suffix)]
        img=nib.load(directory+"\\"+sorted(files)[run-1])
        data=img.get_data()
        return(data)


def coarsen(data):
    assert(len(data.shape)==3)
    coarse=np.zeros(shape=tuple([i//5 for i in data.shape]))
    for it in np.ndindex(coarse.shape):
        x,y,z = it
        coarse[x,y,z]= np.mean(data[5*x:5*(x+1),5*y:5*(y+1),5*z:5*(z+1)])
    return(coarse)

start = time.time()
for i in range(20):
    for j in range(5):
        rd = runData(True, i+1,j+1)
        d = np.asarray([coarsen(rd[:,:,:,i]) for i in range(rd.shape[3])])
        scan = Scan(True, i+1, j+1, d)
        pickle.dump(scan, output, pickle.HIGHEST_PROTOCOL)
        print(i,j)
end = time.time()
print(end-start)

start = time.time()
for i in range(19):
    for j in range(5):
        rd = runData(False, i+1,j+1)
        d = np.asarray([coarsen(rd[:,:,:,i]) for i in range(rd.shape[3])])
        scan = Scan(False, i+1, j+1, d)
        pickle.dump(scan, output, pickle.HIGHEST_PROTOCOL)
        print(i,j)
end = time.time()
print(end-start)