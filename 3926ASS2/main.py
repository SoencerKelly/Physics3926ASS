#Spencer Kelly, 2023
#PHYS3926 Assignment 3
import math

import numpy as np
import matplotlib.pyplot as plt


#part 1
def motion_solver(iSpeed, launchAngle, timeStep, method = 'Midpoint', airRes = True, part3 = False):
    '''Solves equation of motion for a baseball with initial speed and angle using specified method, and removes air resistance if requested. Note: Midpoint method is assumed if no method argument is given (due to highest accuracy).'''
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
    #if air resistance was turned off by argument then return the theoretical value using the equation of motion
    if airRes == False:
        method = 'theoretical'
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
    while currPos[1] >= 0:
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
        #if no air resistance was requested, then return the theoretical value based on the equation of motion
        elif method == 'theoretical':
            nextPos = currPos + currVel*timeStep - 1/2*accel*(timeStep**2)
            nextVel = currVel + accel * timeStep

        #if the function is being used for part 3, return the height value of the ball at the first x value after 400m
        if part3 and currPos[0] >= ft_to_m(400):
            return currPos[1]

        #update the current position for the next iteration of the while loop
        currPos = nextPos
        currVel = nextVel
        #append the current position to the list of positions
        xPos = np.append(xPos, currPos[0])
        yPos = np.append(yPos, currPos[1])
    #return the two position arrays in the case the function is not being used for part 3
    if not part3:
        return xPos, yPos
    #if the function is being used for part 3 and 400m was never reached by the ball before it hit the ground, return a value of -1 as the y pos
    elif part3: return -1

def mph_to_mps(mph):
    '''converts mph to m/s'''
    return mph*0.44704

def ft_to_m(feet):
    '''Converts feet to meters'''
    return 0.3048*feet

#function for part 2
def ABvsHRRats(speed, spd_std, angle, ang_std, length, field_dist, part3 = False, fenceHeight = None):
    '''Creates an array of specified length with a gaussian distribution around the specified speed input with standard deviation of specified parameter
    and an array of same length with a gaussian distribution around the launch angle input parameter with standard deviation of specified input'''
    #create an array for a random gaussian distribution of velocity with standard deviation as specified by parameter
    speedArray = mph_to_mps(speed) + mph_to_mps(spd_std) * np.random.randn(length)
    #create an array for a random gaussian distribution of angles with standard deviation as specified by parameter
    angleArray = angle + ang_std * np.random.randn(length)

    #initialize count variables to count the number of at bats and HR
    ABCount = 0
    HRCount = 0
    #iterate through the two arrays, generating a solution of motion for all initial speeds and their corresponding launch angles
    for iSpeed, iAngle, in zip(speedArray, angleArray):
        ABCount += 1
        if not part3:
            xPos, yPos = motion_solver(iSpeed, iAngle, 0.1, part3 = part3)
            #if the final x position is greater than 400m, add one to the home run counter
            if xPos[-1] >= ft_to_m(field_dist):
                HRCount += 1
        elif part3:
            yPos = motion_solver(iSpeed, iAngle, 0.1, part3 = part3)
            if yPos >= fenceHeight:
                HRCount += 1

    #return the ratio
    try:
        return ABCount/HRCount
    except ZeroDivisionError:
        return str(ABCount) + ' at bats and not a single home run :('

#function for part 3:
def optimal_fence_height():
    '''This function serves to analyze the AB/HR ratio for different fenceheights'''
    #create an array of all fenceheights to be tested
    heightsArray = np.arange(0.5, 15.1, 0.1)
    #create an empty array to append the AB/HR ratios to
    HRArray = np.array([])
    #iterate through each fenceheight
    for i in heightsArray:
        #Calculate the ABvsHR value for the given fenceheight by implementing the part3 argument for the ABvsHR function
        ABvsHR = ABvsHRRats(100, 15, 45, 10, 300, 400, part3=True, fenceHeight=i)
        #append the ratio to the array
        HRArray = np.append(HRArray, ABvsHR)

    return heightsArray, HRArray

def main():
    x, y = motion_solver(50, 45, 0.1, 'Euler')
    x2, y2 = motion_solver(50, 45, 0.1, 'Euler-Cromer')
    x3, y3 = motion_solver(50, 45, 0.1, 'Midpoint')
    print(x,x[-1], y)
    plt.plot(x, y, '--', x2, y2, '+', x3, y3, '-')
    plt.show()
    x,y = motion_solver(15, 45, 0.1, 'Euler')
    x1,y1 = motion_solver(15, 45, 0.1, airRes=False)
    plt.plot(x, y, '--', x1, y1, '+')
    plt.show()
    print(ABvsHRRats(100, 15, 45, 10, 3000, 400))
    y,x = optimal_fence_height()
    plt.plot(x,y)
    plt.show()

main()