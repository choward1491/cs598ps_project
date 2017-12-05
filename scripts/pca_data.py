# import useful libraries
import os
import numpy                as np
import numpy.linalg         as la
import myml.factorizations  as myfac
import myml.images          as myimg
import mysp.sound           as mysnd

# implement main function to be executed
if __name__ == '__main__':

    # specify directory to data
    ddir = '../data'

    # load the data
    if 0:
        D = np.array([])
        for n in range(1,5):
            dir = '{0}/dataset_{1}'.format(ddir,n)
            print('Currently in directory({0})'.format(n))
            for file in os.listdir(dir):
                D0 = np.load('{0}/{1}'.format(dir,file))
                (nt, nx, ny, nz) = D0.shape
                Dt = D0.reshape((nt, nx * ny * nz)).T
                if D.shape[0] == 0:
                    D = np.copy(Dt)
                else:
                    D = np.hstack((D,Dt))
    else:
        D = np.load('../data/raw_features.npy').T

    # get mean of data and subtract it
    (d,nd)  = D.shape
    Dmean   = np.mean(D,axis=1).reshape(d,1)
    Dn      = D - Dmean

    # Get features using randomized range finder algorithm
    (Q,B) = myfac.projrep(Dn,k_or_tol=20)
    print('2-Norm error per matrix element: ',la.norm(Dn - Q@B)/(d*nd))

    # save the resulting data we can use to form the model
    np.save('{0}/pca_features2.npy'.format(ddir),   arr=Q)
    np.save('{0}/pca_weights2.npy'.format(ddir),    arr=B)
    np.save('{0}/pca_mean2.npy'.format(ddir),       arr=Dmean)