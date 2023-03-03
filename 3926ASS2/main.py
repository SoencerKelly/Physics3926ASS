#Spencer Kelly, 2023
#PHYS3926 Assignment 3
import math

import numpy as np
import matplotlib.pyplot as plt

#part 1
def motion_solver(iSpeed, launchAngle, timeStep, method = 'Midpoint', airRes = True, part3 = False, forGraph = False):
    '''Solves equation of motion for a baseball with initial speed and angle using specified method, and removes air resistance if requested. Note: Midpoint method is assumed if no method argument is given (due to highest accuracy).
    Outputs a range in the x direction if not otherwise specified. Is also capable of outputting an array of x and y positions for graphing purposes, and the y value at the length of a baseball field (400ft)'''
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

        #if the function is being used for part 3, return the height value of the ball at the first x value after 400ft
        if part3 and currPos[0] >= ft_to_m(400):
            return currPos[1]

        #update the current position for the next iteration of the while loop
        currPos = nextPos
        currVel = nextVel
        #append the current position to the list of positions
        xPos = np.append(xPos, currPos[0])
        yPos = np.append(yPos, currPos[1])

    #return the range of the ball in the x direction
    if not (part3 or forGraph):
        return xPos[-1]
    # return the two position arrays in the case the function is being used for graphing
    elif forGraph:
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
        #in the case that its not being
        if not part3:
            range = motion_solver(iSpeed, iAngle, 0.1, part3 = part3)
            #if the final x position is greater than 400m, add one to the home run counter
            if range >= ft_to_m(field_dist):
                HRCount += 1
        #if this function is being used for part 3, calculate the AB/HR ratio for the given fenceheight
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
        #only use 300 bats this time, as the computational demand for 3000 in this for loop is too high
        ABvsHR = ABvsHRRats(100, 15, 45, 10, 300, 400, part3=True, fenceHeight=i)
        #append the ratio to the array
        HRArray = np.append(HRArray, ABvsHR)

    #create a curve of best fit using numpy polyfit function
    #start by getting polynomial constants
    a,b,c = np.polyfit(heightsArray, HRArray, 2)
    curve_array = a*heightsArray**2 + b*heightsArray + c


    return heightsArray, HRArray, curve_array

def main():
    #testing out the motion solver function and its methods
    x, y = motion_solver(50, 45, 0.1, 'Euler', forGraph=True)
    x2, y2 = motion_solver(50, 45, 0.1, 'Euler-Cromer', forGraph=True)
    x3, y3 = motion_solver(50, 45, 0.1, 'Midpoint', forGraph=True)
    #plot the 3 sets of points to compare
    plt.plot(x, y, '--', x2, y2, '+', x3, y3, '-')
    plt.show()
    #reproducing figure 2.2 as a test also
    x,y = motion_solver(15, 45, 0.1, 'Euler', forGraph=True)
    x1,y1 = motion_solver(15, 45, 0.1, airRes=False, forGraph=True)
    plt.plot(x, y, '--', x1, y1, '+')
    plt.show()
    #reproduce figure 2.3 as instructed, and save the image
    x, y = motion_solver(50, 45, 0.1, 'Euler', forGraph=True)
    x1, y1 = motion_solver(50, 45, 0.1, airRes=False, forGraph=True)
    plt.plot(x, y, '+', label = 'Euler Method')
    plt.plot(x1, y1, '-', label = 'Theory (No air)')
    plt.title('Projectile motion')
    plt.legend(loc = 'upper left')
    plt.xlabel('Range (m)')
    plt.ylabel('Height (m)')
    plt.savefig('ASS2Fig1')
    plt.show()

    #for part 2: calculate the at bat/home run ratio for the distributions specified in the assignment
    print(ABvsHRRats(100, 15, 45, 10, 3000, 400))

    #for part 3: use the uptimal fence height function to return an array of fence heights and their corresponding AB/HR ratio, as well as a curve of best fit for said set of points
    x,y,curve = optimal_fence_height()

    #for part 3: plot the AB/HR ratio vs fence height array, and the 2nd degree curve of best fit so that an element
    plt.scatter(x, y)
    plt.plot(x, curve, label = 'curve fit')
    plt.xlabel('Fence Height (m)')
    plt.ylabel('At Bats per Home Run')
    #plot a horizontal line to determine the fence height corresponding to an AB/HR ratio of 10
    plt.axhline(10, color='r')
    plt.savefig('ASS2Fig2')
    plt.show()

main()