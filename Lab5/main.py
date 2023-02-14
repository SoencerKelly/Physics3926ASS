##Steven Li and Spencer Kelly, 2023
import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt


def period(max, L):
    '''Finds the period for an undamped unforced oscillator'''
    #Calculate period using equation 2.34 in text
    try:
        integrand = lambda theta: 1/np.sqrt(np.cos(theta) - np.cos(float(max)/57.3))
        T = 4 * (np.sqrt(float(L)/19.6)) * integrate.quad(integrand, 0, float(max)/57.3)[0]
        return T

    #if no input for max angle, assume small angle and use appropriate period formula
    except ValueError:
        return 2 * np.pi * np.sqrt(float(L)/9.8)


def pendulum(t, r, Q, A, w):
    '''Function that returns angular velocity and angular acceleration'''
    #t represents time
    #r represents tuple of anglular displacement and angular velocity
    #Q represents damping constant
    #A force amplitude
    #w is driving force frequency
    #dtheta is the angular velocity
    dtheta = r[1]
    #use equation in assignment notes
    d2theta = (-1/Q) * r[1] - np.sin(r[0]) + A * np.cos(w * t)

    return(dtheta,d2theta)


def event(t, r, Q, A, w):
    return r[1]


def pen_solve(amp, Q, A, w):
    '''uses ivp_solver to solve ivp and returns nested arrays'''
    #amp is maximum amplitude
    #Q represents damping constant
    # A force amplitude
    # t represents time

    #convert
    y=[amp/57.3,0]
    sol = integrate.solve_ivp(pendulum, (0,100), y0=y, t_eval = np.linspace(0,50,100), args = (Q, A, w), events = event)

    x, y = sol.y #x in terms of theta y in terms of dtheta

    t = sol.t #returns all steps

    peaks = sol.t_events[0] #times where angular velocity = 0

    '''create an empty period list:
    if the length of the peaks list is even, then there will need to be (1/2)-1 elements, as there is an odd number as the last index
    if the length of peaks is odd, then simply half the length in modulus 2 will suffice.
    these previous two lines are the reason behind the use of modular arithmetic to specify length for period array'''
    period = np.zeros([1,len(peaks)//2 - 1 + len(peaks)%2])


    # Find time taken between each maxima
    #create count variable to index since the i variable has stepsize 2
    peridx = 0
    for i in range(0,len(peaks) - 2, 2):
        #each period is the distance from one max peak to the next
        perval = peaks[i+2] - peaks[i]
        period[0][peridx] = perval
        peridx +=1

    minmax = np.delete(sol.y_events[0],1,axis = 1) #delete all angular velocities from the tuples returned by y_events (returns tuples [angle, angular velocity]

    return [x * 57.3, y * 57.3 ,t, peaks, period, minmax.transpose()[0]*57.3]


def find_nearest(array, value):
    '''returns index of the element of an array closest to the input value'''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx



def one_over_e(ar):
    '''return time taken to reach amplitude of the max amplitude/e'''
    #extract list of the peak values
    peakvals = ar[5]
    #get index of peak value closest to the maxvalue/e
    decay_time_idx = find_nearest(peakvals, ar[5][0]/np.exp(1))
    #return the time associated with that index
    return ar[2][decay_time_idx]



#create a list of solutions for different starting angles with no damping or driving
undampundriv = [pen_solve(10, float('inf'), 0, 0), pen_solve(30, float('inf'), 0, 0), pen_solve(70, float('inf'), 0, 0)]

#create a list of solutions for various damping constants but no driving constant
damp = [pen_solve(30 ,1 , 0, 0), pen_solve(30 ,5 , 0, 0), pen_solve(30 ,10 , 0, 0), pen_solve(30 ,25 , 0, 0), pen_solve(30 ,50 , 0, 0)]


#solution for both dampened and driven case
dampdriv = pen_solve(30, 20, 0.3, 2)


g, (ax, ax0) = plt.subplots(1, 2)

#Solve for the average and standard deviation of simulation periods for the 3 initial angle cases
av_10 = np.mean(undampundriv[0][4])
std_10 = np.std(undampundriv[0][4])

av_40 = np.mean(undampundriv[1][4])
std_40 = np.std(undampundriv[1][4])

av_70 = np.mean(undampundriv[2][4])
std_70 = np.std(undampundriv[2][4])


#plot the period given by the period function and compare it to the period given by the numerical method from solve_ivp
ax.scatter([10,40,70], [period(10,9.81),period(40,9.81),period(70,9.81)], label = 'Theoretical')
ax.scatter([10,40,70], [av_10, av_40, av_70], label = 'Numerical')
ax.errorbar([10], [av_10], yerr = 1.65*std_10, fmt ='none')
ax.errorbar([40], [av_40], yerr = 1.65*std_40, fmt ='none')
ax.errorbar([70], [av_70], yerr = 1.65*std_70, fmt ='none')
ax.set_ylabel('Max Angular displacement')
ax.set_xlabel('Period')
ax.set_title('Comparing Theoretical vs Numerical Period', fontsize = 9)
ax.legend(loc = 'upper left')


#plot damping constant vs decay time
ax0.scatter([1,5,10,25,50],[one_over_e(damp[0]), one_over_e(damp[1]), one_over_e(damp[2]), one_over_e(damp[3]), one_over_e(damp[4])])
ax0.set_ylabel('decay time (s)')
ax0.set_xlabel('Damping constant')
ax0.set_title('Damping Constant vs Decay Time', fontsize = 9)
g.subplots_adjust(wspace=0.3)

#save figure 2
g.savefig('C:/Users/1spen/Pictures/KellySpencer_Lab5_Fig2.png')



#plot figure 1
plt.clf()
plt.plot(undampundriv[0][2], undampundriv[0][0], undampundriv[1][2], undampundriv[1][0], undampundriv[2][2], undampundriv[2][0])
plt.legend(['10 degrees', '40 degrees', '70 degrees'], loc ="lower right")
plt.title('No damping or driving')
#set line along y=0
plt.plot(np.linspace(0,50,100), np.zeros(100), linestyle = '--', color = 'black')
plt.ylabel('Angular displacement (degrees)')
plt.xlabel('Time (seconds)')
#save figure 1
plt.savefig('C:/Users/1spen/Pictures/KellySpencer_Lab5_Fig1.png')
plt.clf()

#plot fig 3
plt.plot(damp[0][2],damp[0][0],damp[1][2],damp[1][0],damp[2][2],damp[2][0],damp[3][2],damp[3][0],damp[4][2],damp[4][0])
plt.legend(['Q = 1', 'Q = 5', 'Q = 10', 'Q = 25', 'Q = 50'], loc ="lower right")
plt.title('Dampened no Driving')
#set line along y=0
plt.plot(np.linspace(0,one_over_e(damp[4]),100), np.zeros(100), linestyle = '--', color = 'black')
plt.ylabel('Angular displacement (degrees)')
plt.xlabel('Time (seconds)')
#save figure 3
plt.savefig('C:/Users/1spen/Pictures/KellySpencer_Lab5_Fig3.png')
plt.clf()

#plot figure 4
plt.plot(dampdriv[2],dampdriv[0])
#set line along y=0
plt.plot(np.linspace(0,50,100), np.zeros(100), linestyle = '--', color = 'black')
plt.title('Damping and driving')
plt.xlabel('Time (seconds)')
plt.ylabel('Angular displacement (degrees)')
#save figure 4
plt.savefig('C:/Users/1spen/Pictures/KellySpencer_Lab5_Fig4.png')
