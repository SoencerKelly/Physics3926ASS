#Spencer Kelly and Steven Li 2023
import matplotlib.pyplot as plt
from celluloid import Camera
import numpy as np
from scipy import integrate


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

    return [x, t]

def ball_coords(theta):
    '''Returns the x and (negative) y coordinates of the pendulum ball given an angle theta'''
    return np.sin(theta), -np.cos(theta)

def movie_pendulum(t_arr, theta_array):
    '''This functions serves to animate a pendulum swinging, using data acquired from solving the equations of motion'''
    fig, axes = plt.subplots()
    camera = Camera(fig)
    for i in range(len(t_arr)):
        #get pendulum ball coordinates
        x,y = ball_coords(theta_array[i])
        #plot said coordinates
        axes.plot([0,x],[0,y], lw=3, c='b')
        #add the ball
        axes.add_patch(plt.Circle(ball_coords(theta_array[i]), 0.1,fc='b', zorder=3))
        #add text to read the angle and time for all given array elements in the animation
        theta_txt = 'Angle (degrees) = ' + str(57.3*theta_array[i])
        time_txt = 'Time (s) = ' + str(t_arr[i])
        plt.text(0,0, theta_txt)
        plt.text(0,-1, time_txt)
        #Add a title specifying the pendulum's simulation parameters
        axes.set_title('Pendulum for Starting angle = 30 degrees, driving force frequency = 2 rad/s, Q = 20, Force Amplitude = 0.2', fontsize=7)
        camera.snap()
    #assing the animation to a variable
    animation = camera.animate()
    #save the animation
    animation.save('SpencerKelly_Lab6_pendulum.gif', writer='PillowWriter', fps=10)



def main():
    dampdriv = pen_solve(30, 20, 0.3, 2)
    print(len(dampdriv[1]), len(dampdriv[0]))
    movie_pendulum(dampdriv[1], dampdriv[0])

main()