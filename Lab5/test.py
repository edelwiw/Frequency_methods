import numpy as np
import matplotlib.pyplot as pl

#Consider function f(t)=1/(t^2+1)
#We want to compute the Fourier transform g(w)

#Discretize time t
t0=-100
t = np.linspace(t0, -t0, 10000)
dt = t[1] - t[0]
#Define function
get_wave_func = lambda a, t1, t2: np.vectorize(lambda t: a if t1 <= t <= t2 else 0, otypes=[complex])
f = get_wave_func(1, -0.5, 0.5)(t)

#Compute Fourier transform by numpy's FFT function
g=np.fft.fft(f)
#frequency normalization factor is 2*np.pi/dt
w = np.fft.fftfreq(f.size)*2*np.pi/dt
print(w[-100:])

#In order to get a discretisation of the continuous Fourier transform
#we need to multiply g by a phase factor
g*=dt*np.exp(-complex(0,1)*w*t0)/(np.sqrt(2*np.pi))

#Plot Result
pl.plot(w,g,color="r")
#For comparison we plot the analytical solution
pl.plot(w,np.sinc(w),color="g")

pl.gca().set_xlim(-30,30)
pl.show()
pl.close()