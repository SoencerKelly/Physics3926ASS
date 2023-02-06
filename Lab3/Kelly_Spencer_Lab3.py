#The code below is written for Lab3 by Spencer Kelly
import numpy as np
import matplotlib.pyplot as plt
def physconst(const_string):
    '''This function serves to convert a string signifying a constant
     to the constants value'''
    const_string = const_string.lower()
    const_dictionary = {
        'c': 2.99792458*(10**8),
        'h': 6.62606896*(10**(-34)),
        'k': 1.3806504*(10**-23),
        'Me': 9.10938215*(10**-31),
        'amu': 1.660538782*(10**-27),
        'e': 1.602176487*(10**-19),
        'a0': 5.2917720859*(10**-11),
        'g': 6.67428*(10**-11),
        'sigma': 5.670400*(10**-8)

    }
    return const_dictionary[const_string]

def planck(v,T):
    '''Calculates the planck radiation for given temperatures and frequencies'''
    #first convert the wavelength to

    #for v, when plugging into B, shange its axis so that it is a column vector and can be stretched in conjunction with T for broadcasting
    B = 2*(physconst('h') * (v**3)/(physconst('c')**2)) * (np.exp(physconst('h') * v/(physconst('k')* T[:, np.newaxis])))**-1
    return B

def planckPlotter(radiation, xval):
    '''PLots the radiation vs the frequency for 3 seperate temperatures'''
    plt.plot(physconst('c')/xval, radiation[0], label = '5000K')
    plt.plot(physconst('c')/xval, radiation[1], label = '10000K')
    plt.plot(physconst('c') / xval, radiation[2], label='15000K')
    plt.xlabel('Wavelength (m)')
    plt.ylabel('Radiation')
    plt.title('Planck Radiation vs Wavelength')
    plt.legend()
    plt.savefig('C:/Users/1spen/Pictures/KellySpencer_Lab3_Fig1.png')
    plt.show()

#Main function where all code is run from
def main():
    #create array with all wavelengths from 10 to 2000nm
    v = np.linspace(3*10**16, 1.5*10**14, 1000)
    #create array of 3 temperatures to graph
    T = np.array([5000, 10000, 15000])
    #input the two previous arrays into the planck function
    B = planck(v, T)
    #now plug B and our frequency array into planckPlotter
    planckPlotter(B, v)
    #now calculate a new array containing the integrals of our 3 planck functions
    trapz_list = np.array([np.trapz(B[0], -v), np.trapz(B[1], -v), np.trapz(B[2], -v)])
    #Create array for the 3 Stefan-Boltzmann calculations for each temp
    boltzmann_list = []
    for temp in T:
        #plug in temperature to the Stefan-Boltzmann equation
        #note: temperature must be converted to floating point number before being put to the power of 4 otherwise it produces seemingly meaningless numbers
        boltzmann_list.append((physconst('sigma')*float(temp)**4)/np.pi)
    #plot the two lists to see if their graph is comparable to that of y=x
    plt.plot(boltzmann_list, trapz_list)
    plt.xlabel('Stefan-Boltzmann Value')
    plt.ylabel('Planck Integration Value')
    plt.title('Comparing Stefan-Boltzmann values to Planck\'s')
    plt.savefig('C:/Users/1spen/Pictures/KellySpencer_Lab3_Fig2.png')
    plt.show()



main()