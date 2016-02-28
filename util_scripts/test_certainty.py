import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.special import erf, erfinv

p = np.linspace(0.0, 1.0, 250)

pp = ((1-(1-p)**5) + (p**5))/2

fig = plt.figure()
mngr = plt.get_current_fig_manager()
# to put it into the upper left corner for example:
mngr.window.setGeometry(50, 100, 600, 300)

im1 = fig.gca().plot(p, pp)

def init():
    im1[0].set_ydata(pp)

def animate(i):
    param = np.exp((float(i)-50)/10)
    #pp = ((1-(1-p)**param) + (p**param))/2

    pp = (erf( erfinv( p*2-1 ) * param )+1)/2

    fig.suptitle("power %f"%float(param))
    im1[0].set_ydata(pp)
    return im1

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=100, interval=50)
#anim.save('my_animation.mp4')
plt.show()