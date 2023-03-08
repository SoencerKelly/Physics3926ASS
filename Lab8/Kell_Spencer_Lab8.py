import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
def FTCS_solva(nspace, ntime, tau_rel, args):
    # * Initialize parameters (time step, grid spacing, etc.).
    N = nspace
    L = args[0]  # The system extends from x=-L/2 to x=L/2
    h = L / (N - 1)  # Grid size
    tau = tau_rel * h**2/(2*args[1])
    coeff = args[1] * tau / h ** 2
    if coeff < 0.5:
        print('Solution is expected to be stable')
    else:
        print('WARNING: Solution is expected to be unstable')

    # * Set initial and boundary conditions.
    tt = np.zeros(shape = (ntime,N))  # Initialize temperature to zero at all points
    tt[0][int(N / 2)] = 1. / h  # Initial cond. is delta function in center
    ## The boundary conditions are tt[0] = tt[N-1] = 0

    # * Set up loop and plot variables.
    xplot = np.arange(N) * h - L / 2.  # Record the x scale for plots
    tplot = np.arange(ntime) * tau

    # * Loop over the desired number of time steps.
    for istep in range(0, ntime - 1):  ## MAIN LOOP ##

        # * Compute new temperature using FTCS scheme.
        #use the current row to figure out temp distribution in next row
        tt[istep+1][1:(N - 1)] = (tt[istep][1:(N - 1)] +
                         coeff * (tt[istep][2:N] + tt[istep][0:(N - 2)] - 2 * tt[istep][1:(N - 1)]))

    return tt, xplot, tplot
plot, xvals, tvals = FTCS_solva(61, 300, 0.72, [1,1])

plt.contour(xvals, tvals, plot, range(1,11))
plt.show()
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
plot = plot[5:]
tvals = tvals[5:]


x,y = np.meshgrid(xvals, tvals)
ax.plot_surface(x,y,plot, rstride=2, cstride=2, cmap=cm.gray)
plt.show()