#Spencer Kelly, 2023

import numpy as np
import scipy as sc
import astropy.units as u
import matplotlib.pyplot as plt
import astropy.constants as c
import pandas as pd



#define functions to be solved
def funcdr(r, rho_m_tuple, tol = 1e-11):
    '''The purpose of this function is to act purely as a mathematical function defining the ODEs drho/dr and dm/dr
    and returning their values given the input rho, m and r
    Args:
        r (float): parameter representing radius; distance from center of star
        rho_m_tuple (tuple): tuple containing floating point values for:
            rho (float): density
            m (float): mass
            '''
    #unpack rho and m from the tuple
    rho, m = rho_m_tuple

    #define x to be rho^(1/3)
    x = rho**(1/3)

    #define gammax
    gammax = x**2/(3*(1+x**2)**(1/2))

    #define drho/dr
    drhodr = -(m*rho)/(gammax*r**2)

    #define dm/dr
    dmdr = r**2*rho

    return drhodr, dmdr


#Write small function to stop integration when the t array reaches a low enough value (10^-11)
def integrationStopper(r, rho_m_tuple, tol = 1e-11):
    return rho_m_tuple[0] - tol


def ODESolver(rhoc, method ='RK45'):
    '''Uses scipy's solve_ivp function to solve a set of ODEs. Stops integrating when the density drops to
     approximately 0 (10^-11) and returns the last element of mass and radius from the solution list
     Args:
        rhoc (float): Central density of star
        method (string): Method in which solve_ivp should use for numerical integration.'''

    #use solve_ivp to solve initial value problem from part 1
    sol = sc.integrate.solve_ivp(funcdr, (1e-11, 100), (rhoc, 0), method = method, events = integrationStopper)

    #find index where the density drops to (approximately) 0 and assign the radius of the star to the corresponding index in the list of radii
    rs = sol.t[-1]
    #find the corresponding mass of the star
    m = sol.y[1][-1]

    #return the tuple of mass and radius
    return (m, rs)


def quantifyUnits(radius_mass_array):
    '''Converts list of tuples into units of M_sun and R_sun
    Args:
        radius_mass_array (array): List of tuples to be converted'''
    #define R_0 and M_0 using values provided and taking mu_e = 2
    #using astropy units to
    R_0 = 7.72e8/2 * u.cm
    M_0 = 5.67e33/2**2 * u.g

    R_0 = R_0.to(u.km)
    M_0 = M_0.to(u.kg)
    #now we want them in terms of sun radius and mass
    R_0 = R_0 /c.R_sun.to(u.km)
    M_0 = M_0 /c.M_sun

    #multiply the radius and mass list by the reference values to create a radius and mass list with dimensions for graphing
    dim_mass_array = radius_mass_array[:,0] * M_0
    dim_rad_array = radius_mass_array[:,1] * R_0

    #return the array with dimensions
    return dim_mass_array, dim_rad_array


def main():
    integrationStopper.terminal = True
    #take 10 values each on a seperate order of magnitude for rhoc between 10^-1 and 10^5to be used for calculation
    rhoc_array = np.logspace(-1, 5, 9)

    # the last value is supposed to be 2.5*10^6 but we need to add that seperately as it isnt possible from the logspace function.
    rhoc_array = np.append(rhoc_array, 2.5e6)

    #create an array to store the values for radius and mass from the calculations
    mass_radius_list = []

    #iterate through all the rhoc values and append the mass and radius to our list
    for val in rhoc_array:
        m, rs = ODESolver(val)
        mass_radius_list.append([m,rs])

    #now that we have a 2D list, we want to convert it to a numpy array for ease of calculation
    mass_radius_list = np.array(mass_radius_list)

    #now use the quantify_units function to give our list of radii and masses in terms of solar radii and mass
    dimension_array = quantifyUnits(mass_radius_list)

    #plot the results
    plt.plot(dimension_array[:][0], dimension_array[:][1])
    plt.xlabel('Mass (Msun)')
    plt.ylabel('Radius (Rsun)')
    plt.title('Mass vs Radius')
    plt.savefig('ASS3Fig1.png')
    plt.show()


    #pick last 3 values from the rhoc array and solve them again using a different method (we will use DOP853) in the solve_ivp function
    new_rhoc_array = np.array([rhoc_array[0], rhoc_array[4], rhoc_array[9]])
    new_mass_radius_list = []
    for val in new_rhoc_array:
        m, rs = ODESolver(val, method='DOP853')
        new_mass_radius_list.append([m,rs])

    #convert our list to an array
    new_mass_radius_list = np.array(new_mass_radius_list)

    #now we compare the values of our original solution (which uses Runge Kutta 5th order) with that of the DOP853 method
    first_mass_rat = new_mass_radius_list[0][0]/mass_radius_list[0][0]
    second_mass_rat = new_mass_radius_list[1][0]/mass_radius_list[4][0]
    third_mass_rat = new_mass_radius_list[2][0]/mass_radius_list[9][0]
    print('The ratio of masses from the second/first method are:', first_mass_rat, second_mass_rat, third_mass_rat)

    first_radius_rat = new_mass_radius_list[0][1]/mass_radius_list[0][1]
    second_radius_rat = new_mass_radius_list[1][1]/mass_radius_list[4][1]
    third_radius_rat = new_mass_radius_list[2][1]/mass_radius_list[9][1]
    print('The ratio of radii from the second/first method are:', first_radius_rat, second_radius_rat, third_radius_rat)


    #now we want to read in data from the csv file provided and compare it with our theoretical solutions
    df = pd.read_csv('wd_mass_radius.csv')

    #now we will plot it against our theoretical values from the first graph
    plt.plot(dimension_array[:][0], dimension_array[:][1], label='Theoretical Values')
    plt.scatter(df['M_Msun'], df['R_Rsun'], label = 'Tremblay et al.', color='orange')
    plt.errorbar(df['M_Msun'], df['R_Rsun'], xerr=df['M_unc'], yerr=df['R_unc'], linestyle='')
    plt.xlabel('Mass (Msun)')
    plt.ylabel('Radius (Rsun)')
    plt.legend()
    plt.title('Mass vs Radius')
    plt.savefig('ASS3Fig2.png')
    plt.show()

main()
