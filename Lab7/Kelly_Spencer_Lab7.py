import numpy as np
import scipy
import matplotlib.pyplot as plt


def determinant_calc():
    # create set of random 4x4 arrays with values from 0 to 1
    setArray = np.array(
        [np.random.rand(4, 4), np.random.rand(4, 4), np.random.rand(4, 4), np.random.rand(4, 4), np.random.rand(4, 4)])
    # calculate determinants using scipy and numpy
    dets = np.linalg.det(setArray)
    # initialize empty list of determinants for scipy
    dets2 = []
    # iterate through all the elements and calculate the determinant
    print(setArray[1])
    for i in range(len(setArray)):
        dets2.append(scipy.linalg.det(setArray[i]))
    print(dets, dets2)
    # compare the methods
    for det1, det2 in zip(dets, dets2):
        print(det1 - det2)
    # all super small difference

def matrix_constructor(ki, Li):
    '''takes 2 1-D arrays for ki and Li coeffs and creates matrix, where Lw is to be passed as the 5th element in the Li list'''
    #create matrix to represent system of linear equations
    systemEqn = np.array([[-ki[0]-ki[1], ki[1], 0],
              [ki[1], -ki[1]-ki[2], ki[2]],
              [0, ki[2], -ki[2]-ki[3]]])
    #create 1-D array to represend constants
    bTermsArray =  np.array([ki[0]*Li[0] - ki[1]*Li[1], ki[1]*Li[1]- ki[2]*Li[2], ki[2]*Li[2]-ki[3]*Li[3] - ki[3]*Li[4]])
    return systemEqn, bTermsArray

def my_function(t, y, ki, Li, mi):
    #assign values from state vector
    x = [y[0], y[1], y[2]]
    dx1 = y[3]
    dx2 = y[4]
    dx3 = y[5]
    #create coefficent matrix and constant array using matrix_constructor
    syst, const = matrix_constructor(ki, Li)
    #calculate the dv/dt values based off equations given in assignment
    dv1 = (np.matmul(syst,np.array(x)) - const[0]/mi[0])[0]
    dv2 = (np.matmul(syst,np.array(x)) - const[1]/mi[1])[1]
    dv3 = (np.matmul(syst,np.array(x)) - const[2]/mi[2])[2]
    #return the dx and dv values
    return dx1,dx2,dx3,dv1,dv2,dv3

def main():
    determinant_calc()
    #create lists of ki and Li
    ki = [1,2,3,4]
    Li = [1,1,1,1,10]
    #create a matrix
    matrix, bterms = matrix_constructor(ki,Li)
    #solve the matrix
    xvals1 = np.linalg.solve(matrix, bterms)
    print(xvals1)
    #create second set of lists for ki and Li
    ki = [1, 1, 1, 1]
    Li = [2, 2, 1, 1, 4]
    # create a matrix
    matrix, bterms = matrix_constructor(ki, Li)
    # solve the matrix
    xvals2 = np.linalg.solve(matrix, bterms)
    print(xvals2)
    #set mass to 1 for all
    mi = np.array([1, 1, 1])
    #create initial condtion array for solve_ivp
    init_cond = np.concatenate((xvals1[:3], np.zeros(3)))
    #reset the ki and Li values
    ki = [1, 2, 3, 4]
    Li = [1, 1, 1, 1, 10]
    sol = scipy.integrate.solve_ivp(my_function, t_span=(0,500), t_eval = np.linspace(0,500,1000), args = (ki,Li,mi), y0 = init_cond)
    x1,x2,x3,v1,v2,v3 = sol.y
    t = sol.t
    print (x1)


    #now we begin graphing
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(t,x1)
    ax2.plot(t,x2)
    ax3.plot(t, x3)
    plt.show()
    #this makes sense as there is no friction so the force from one weight on another can only increase, so there is more and more displacement with every oscillation

    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(t, v1)
    ax2.plot(t, v2)
    ax3.plot(t, v3)
    plt.show()
    # This makes sense, because as mentioned above, there is no friction so the increased force each oscillation increases each successive velocity in turn

    # reset the ki and Li values to the second pair
    ki = [1, 1, 1, 1]
    Li = [2, 2, 1, 1, 4]
    sol = scipy.integrate.solve_ivp(my_function, t_span=(0, 500), t_eval=np.linspace(0, 500, 1000), args=(ki, Li, mi),
                                    y0=init_cond)
    x1, x2, x3, v1, v2, v3 = sol.y
    t = sol.t
    print(x1)

    # now we begin graphing
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(t, x1)
    ax2.plot(t, x2)
    ax3.plot(t, x3)
    plt.show()
    # this makes sense as there is no friction so the force from one weight on another can only increase, not decrease

    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(t, v1)
    ax2.plot(t, v2)
    ax3.plot(t, v3)
    plt.show()
    # This makes sense, because as mentioned above, there is no friction so the oscillations can't decrease in amplitude


main()
