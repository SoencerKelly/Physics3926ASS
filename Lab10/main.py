#Spencer Kelly
#Lab 8
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt


def make_tridiagonal(N,b,d,a):
    '''Creates a tridiaonal matrix of Nth dimension with input parameter b being the lower diagonal,
    d being along the diagonal, and a being the diagonal above d'''
    #use scipy.sparse.diags to create the desired tridiagonal matrix
    tridag_array = sc.sparse.diags([b,d,a], [-1,0,1], shape=(N,N)).toarray()
    print(tridag_array)

def make_initialcond(o0, k0, xi):
    '''returns equation for initial wavepacket given input parameters'''
    return np.exp((-xi**2)/(2*o0**2)) * np.cos(k0*xi)

def spectral_radius(array):
    '''returns the highest magnitude eigenvalue of the input array'''
    vals, vecs = np.linalg.eig(array)
    #find element with largest absolute value
    maxeig = vals[abs(vals) == max(abs(vals))]

    #return the integer value of the single item list in maxeig
    return int(maxeig)

#test make_initialcond function:
xi = np.linspace(-2.5,2.5,300)
plt.plot(xi, make_initialcond(0.2, 35, xi))
plt.ylabel('a(x,0)')
plt.xlabel('x')
plt.show()
plt.savefig('KellySpencer_Lab10_Fig1.png')

#test spectral_radius function
print(spectral_radius(np.diag((1,2,3))))

make_tridiagonal(5,3,1,5)