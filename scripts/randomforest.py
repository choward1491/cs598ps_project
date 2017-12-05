# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:15:21 2017

@author: manchan2
"""

from dnn_regression import PatientDataset, TimeCrop
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import os
import pydot

if __name__=="__main__":    
    dataset = PatientDataset('C:\\Users\\schan\\Documents\\LargeFiles\\nmf.pkl', transform = TimeCrop(68, random=False))
    print('Length of Dataset: ', len(dataset))
    features=dataset.getTrainFeatures()
    labels=dataset.getTrainLabels()
    rf = RandomForestClassifier()
    rf.fit(features, labels)
    
    valFeat = dataset.getValFeatures()
    valLab = dataset.getValLabels()
    
    pred = rf.score(valFeat,valLab)
    print(pred)
    print(rf.predict_proba(valFeat))
    
    export_graphviz(rf.estimators_[0])
    os.system('dot -Tpng tree.dot -o tree.png')
    
    (graph,) = pydot.graph_from_dot_file('tree.dot')
    graph.write_png('tree.png')