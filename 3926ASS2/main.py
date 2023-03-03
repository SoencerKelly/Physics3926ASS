#Spencer Kelly, 2023
#PHYS3926 Assignment 3
import math

import numpy as np
import matplotlib.pyplot as plt


#part 1
def motion_solver(iSpeed, launchAngle, timeStep, method = 'Euler', airRes = True):
    '''Solves equation of motion for a baseball with initial speed and angle using specified method, and removes air resistance if requested'''
    #List all constants used in the calculation:
    #according to the textbook, we can approximate the drag coefficient as 0.35
    C_d = 0.35
    #Density of air is taken to be 1.2kg/m^3
    airDen = 1.2
    #baseball diameter is 7.4 cm, so the cross-sectional area is...
    A = np.pi * (0.074/2)**2
    #earth's gravity
    g = 9.81
    #the mass of the ball
    m = 0.145
    #set air resistance to 0 if requested by the user inputting the parameters
    if airRes == False:
        airRes = 0
    else:
        airRes = C_d * airDen * A

    #Now, we begin to consider initial conditions...
    #The initial position is set to be (x,y) = (0,1)
    currPos = np.array([0,1])
    #create an array for x coordinates and for y coordinates
    xPos = np.array(currPos[0])
    yPos = np.array(currPos[1])
    #initialize the velocity in 2D space
    currVel = np.array([iSpeed * np.cos(math.radians(launchAngle)), iSpeed*np.sin(math.radians(launchAngle))])
    #create a while loop to implement the Euler method until the ball hits the ground
    while currPos[1] > 0:
        # accel is the net acceleration on the object, and is the sum of the drag force divided by the mass and gravitational
        accel = (-1 / 2 * airRes * math.sqrt(currVel[0]**2 + currVel[1]**2))/m *currVel -  g*np.array([0,1])
        if method == 'Euler':
            # the next position is a function of the current velocity
            nextPos = currPos + timeStep * currVel
            #the next velocity is dependent on acceleration and the current velocity
            nextVel = currVel + timeStep * accel

        elif method == 'Euler-Cromer':
            #the Euler Cromer calculates the next position using the next velocity as opposed to the current
            nextVel = currVel + timeStep * accel
            nextPos = currPos + timeStep * nextVel
        elif method == 'Midpoint':
            #the midpoint method uses the average of the next velocity and the current velocity to calculate the next position
            nextVel = currVel + timeStep * accel
            nextPos = currPos + timeStep * ((nextVel + currVel)/2)





        currPos = nextPos
        currVel = nextVel
        xPos = np.append(xPos, currPos[0])
        yPos = np.append(yPos, currPos[1])
    return xPos, yPos

#part 2
def mph_to_mps(mph):
    '''converts mph to m/s'''
    return mph*0.44704

def ABvsHRRats(speed, spd_std, angle, ang_std, length):
    '''Creates an array of specified length with a gaussian distribution around the specified speed input with standard deviation of specified parameter
    and an array of same length with a gaussian distribution around the launch angle input parameter with standard deviation of specified input'''
    #create an array for a random gaussian distribution of velocity with standard deviation as specified by parameter
    speedArray = mph_to_mps(speed) + spd_std * np.random.randn(length)
    #create an array for a random gaussian distribution of angles with standard deviation as specified by parameter
    angleArray = angle + ang_std * np.random.randn(length)

    #initialize count variables to count the number of at bats and HR
    ABCount = 0
    HRCount = 0
    #iterate through the two arrays, generating a solution of motion for all initial speeds and their corresponding launch angles
    for iSpeed, iAngle, in zip(speedArray, angleArray):
        xPos, yPos = motion_solver(iSpeed, iAngle, 0.1, 'Euler')
        ABCount += 1
        if xPos[-1] >= 400:
            HRCount += 1
    return ABCount/HRCount



def main():
    x, y = motion_solver(50, 45, 0.1, 'Euler')
    x2, y2 = motion_solver(50, 45, 0.1, 'Euler-Cromer')
    x3, y3 = motion_solver(50, 45, 0.1, 'Midpoint')
    print(x, y)
    plt.plot(x, y, '--', x2, y2, '+', x3, y3, '-')
    plt.show()
    ABvsHRRats(100, 15, )