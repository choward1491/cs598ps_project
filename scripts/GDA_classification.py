
#import useful libraries
import numpy                as np
import myml.discriminant    as mydsc
import myml.data_separation as mydata
import myml.images          as myimg
import myml.factorizations  as myfac
import matplotlib.pyplot    as plot
import time
from mpl_toolkits.mplot3d import Axes3D


def gda_model_accuracy(Xraw, L, num_features, percent_train=0.5):

    # reduce dimensions of raw data
    (Q, X) = myfac.projrep(Xraw, k_or_tol=num_features)

    # get shape information for labels
    (nl,) = L.shape

    # break data into training and testing datasets
    (Dtr, Dtt) = mydata.separateClassData(X, L.reshape(1, nl), numdata_or_percent_for_training=percent_train)


    # get the start time the training starts
    start = time.time()

    # build the set of LDA models
    dlist = mydsc.constructDistriminantSet(X=Dtr['net'], L=Dtr['nlbl'])

    # get the end time after training
    end = time.time()
    print('({0}) Time elapsed to train is: '.format(num_features),end-start)

    # test the discriminants for each label dataset in the training set
    Ltest = mydsc.evalDiscriminantSet(X=Dtt['net'], discriminant_list=dlist)

    # Compute accuracy and confusion matrix
    classes = np.array([0, 1])
    (C, accuracy) = mydsc.computeConfusionMatrix(Xtest=Dtt['net'], Ltest=Dtt['nlbl'], Leval=Ltest, classes=classes)

    # plot the confusion matrix
    f0 = plot.figure()
    (f0, ax0) = myimg.plotDataset(f0, C, delta=(-1, -1))
    ax0.set_xlabel('Classified Classes')
    ax0.set_ylabel('True Classes')
    f0.savefig('{0}/confusion_mat_{1}.png'.format('../plots/GDA',num_features))

    # print output message
    print('Accuracy for {0} features is : '.format(num_features), accuracy)

    # return the accuracy
    return accuracy/100.0

if __name__ == '__main__':

    # load the raw inputs and labels
    Xraw0 = np.load('../data/raw_features.npy').T
    L = np.load('../data/control_lbls.npy')

    # get mean of input data and subtract it
    doWhiten = False
    if doWhiten:
        (d, nd) = Xraw0.shape
        Xmean = np.mean(Xraw0, axis=1).reshape(d, 1)
        Xraw = Xraw0 - Xmean
    else:
        Xraw = Xraw0

    # loop through different number of feature representations
    # to see what the overall classification accuracy is for each
    klist = np.arange(2,24,2)
    alist = np.zeros(klist.shape)
    for i in range(0,klist.shape[0]):
        alist[i] = gda_model_accuracy(Xraw,L,klist[i])

    # plot the results
    fig = plot.figure()
    plot.plot(klist,alist)
    plot.xlabel('Number of Features')
    plot.ylabel('Testing Classification Accuracy')
    fig.savefig('{0}/fdim_vs_accuracy.png'.format('../plots/GDA'))
