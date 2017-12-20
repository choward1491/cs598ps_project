#import useful libraries
import numpy                    as np
import matplotlib.pyplot        as plot


if __name__ == '__main__':

    # load PCA and NMF accuracy data for GDA
    a_gpca = 100*np.load('../data/gpca.npy')
    a_gnmf = 100*np.load('../data/gnmf.npy')

    # load PCA and NMF accuracy data for Random Forests
    a_rfpca = 100*np.load('../data/rfpca.npy')
    a_rfnmf = 100*np.load('../data/rfnmf.npy')

    # load PCA and NMF accuracy data for Deep Neural Network
    a_dpca = 100*np.load('../data/dfpca.npy')
    a_dnmf = 100*np.load('../data/dfnmf.npy')

    # feature size values
    k = np.arange(1,21,dtype=int)

    # plot the results
    fig1 = plot.figure()
    l1, = plot.plot(k, a_gpca, label='GDA - PCA', color=(0.7, 0, 1.0, 1.0), linewidth=2)
    l2, = plot.plot(k, a_gnmf, label='GDA - NMF',color=(0.7, 0, 1.0, 1.0), linewidth=2, linestyle='dashed', marker='^')
    l3, = plot.plot(k, a_rfpca, label='RF - PCA',color=(0, 1.0, 0.3, 1.0), linewidth=2)
    l4, = plot.plot(k, a_rfnmf, label='RF - NMF',color=(0, 1.0, 0.3, 1.0), linewidth=2, linestyle='dashed', marker='^')
    l5, = plot.plot(k, a_dpca, label='DNN - PCA',color=(1.0, 0.6, 0, 1.0), linewidth=2)
    l6, = plot.plot(k, a_dnmf, label='DNN - NMF',color=(1.0, 0.6, 0, 1.0), linewidth=2, linestyle='dashed', marker='^')
    plot.xlabel('Number of Features')
    plot.ylabel('Testing Classification Accuracy (%)')
    plot.xticks(k)
    plot.legend(handles=[l1,l2,l3,l4,l5,l6])
    fig1.savefig('{0}/fdim_vs_accuracy_all.png'.format('../plots'))