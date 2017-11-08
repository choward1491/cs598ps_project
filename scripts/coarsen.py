# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 10:20:25 2017

@author: manchan2
"""


import numpy as np
import nibabel as nib
import os
#import matplotlib.pyplot as plt
#from sklearn.decomposition import NMF 
import time
def runData(subj, run):
    suffix="_bold.nii.gz"
    directory = "C:\\Users\\schan\\OneDrive\\Documents\\Homework\\Fall 2017\\MLSP\\project\\ds171_R1.0.0_controls\\ds171_R1.0.0_controls\\ds171_R1.0.0\\sub-control"+str(subj).zfill(2)+"\\func"
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
        rd = runData(i+1,j+1)
        d = np.asarray([coarsen(rd[:,:,:,i]) for i in range(rd.shape[3])])
        np.save("subj"+str(i+1).zfill(2)+"_task"+str(j+1)+".npy",d)
        print(i,j)
end = time.time()
print(end-start)
