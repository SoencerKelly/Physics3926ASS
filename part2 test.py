import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt


def period(max, L):
    try:
        integrand = lambda theta: 1/np.sqrt(np.cos(theta) - np.cos(float(max)/57.3))
        T = 4 * (np.sqrt(float(L)/19.6)) * integrate.quad(integrand, 0, float(max)/57.3)[0]
        return T
    
    except ValueError:
        return 2 * np.pi * np.sqrt(float(L)/9.8)


def pendulum(t, r, Q, A, w):
    dx = r[1]
    d2x = (-1/Q) * r[1] - np.sin(r[0]) + A * np.cos(w * t)

    return(dx,d2x)


def event(t, y, Q, A, w):
    return y[1]


def pen_solve(amp, Q, A, w):
    y=[amp/57.3,0]
    sol = integrate.solve_ivp(pendulum, (0,100), y0=y, t_eval = np.linspace(0,50,100), args = (Q, A, w), events = event)

    x, y = sol.y
    t = sol.t
    zeros = sol.t_events[0]
    period = zeros[2]
    minmax = np.delete(sol.y_events[0],1,axis = 1)

    return [x * 57.3, y * 57.3 ,t, zeros, period, minmax.transpose()[0]]


undampdriv = [pen_solve(10,float('inf'), 0, 0), pen_solve(30,float('inf'), 0, 0), pen_solve(70,float('inf'), 0, 0)]
damp = [pen_solve(30 ,1 , 0, 0), pen_solve(30 ,5 , 0, 0), pen_solve(30 ,10 , 0, 0), pen_solve(30 ,25 , 0, 0), pen_solve(30 ,50 , 0, 0)]
dampdriv = pen_solve(30, 20, 0.3, 2)


g, (ax, ax0) = plt.subplots(1, 2)

ax.scatter([10,40,70], [period(10,9.81),period(40,9.81),period(70,9.81)])
ax.scatter([10,40,70], [undampdriv[0][4],undampdriv[1][4],undampdriv[2][4]])
ax.errorbar([10,40,70], [undampdriv[0][4],undampdriv[1][4],undampdriv[2][4]], yerr = 0.1, fmt = 'none')

print(damp[0][3])
print(np.log(np.abs(damp[4][5])*57.3) - np.log(30))
ax0.scatter([1,5,10,25,50],[(np.log(np.abs(damp[0][3])*57.3) - np.log(30))[1], (np.log(np.abs(damp[1][3])*57.3) - np.log(30))[3], (np.log(np.abs(damp[2][3])*57.3) - np.log(30))[6], (np.log(np.abs(damp[3][3])*57.3) - np.log(30))[16], (np.log(np.abs(damp[4][3])*57.3) - np.log(30))[31]])
#ax0.scatter([1,5,10,25,50],[(np.log(np.abs(damp[0][5])*57.3) - np.log(30))[1], (np.log(np.abs(damp[1][5])*57.3) - np.log(30))[3], (np.log(np.abs(damp[2][5])*57.3) - np.log(30))[6], (np.log(np.abs(damp[3][5])*57.3) - np.log(30))[16], (np.log(np.abs(damp[4][5])*57.3) - np.log(30))[31]])



f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

ax1.plot(undampdriv[0][2],undampdriv[0][0], undampdriv[1][2],undampdriv[1][0], undampdriv[2][2],undampdriv[2][0])
ax1.legend(['10 degrees', '40 degrees', '70 degrees'], loc ="lower right")

ax2.plot(damp[0][2],damp[0][0],damp[1][2],damp[1][0],damp[2][2],damp[2][0],damp[3][2],damp[3][0],damp[4][2],damp[4][0])
ax2.legend(['Q = 1', 'Q = 5', 'Q = 10', 'Q = 25', 'Q = 50'], loc ="lower right")

ax3.plot(dampdriv[2],dampdriv[0])

ax1.plot(np.linspace(0,10,100), np.zeros(100), linestyle = '--', color = 'black')
ax2.plot(np.linspace(0,10,100), np.zeros(100), linestyle = '--', color = 'black')
ax3.plot(np.linspace(0,10,100), np.zeros(100), linestyle = '--', color = 'black')

ax1.set_title('No damping or driving')
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Angular displacement (degrees)')

ax2.set_title('Damping only')
ax2.set_xlabel('Time (seconds)')

ax3.set_title('Damping and driving')
ax3.set_xlabel('Time (seconds)')

plt.show()



