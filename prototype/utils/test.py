import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

total = 10

num_whatever = 100

to_plot = [np.random.rand(num_whatever, 3) for i in range(total)]
colors = [np.tile([0,1],num_whatever//2) for i in range(total)]
red_patch = mpatches.Patch(color='red', label='Men')
blue_patch = mpatches.Patch(color='blue', label='Women')

fig = plt.figure()
ax3d = Axes3D(fig)
scat3D = ax3d.scatter([],[],[], s=10, cmap="bwr", vmin=0, vmax=1)
scat3D.set_cmap("bwr") # cmap argument above is ignored, so set it manually
ttl = ax3d.text2D(0.05, 0.95, "", transform=ax3d.transAxes)


def update_plot(i):
    ttl.set_text('PCA on 3 components at step = {}'.format(i*20))
    test = np.transpose(to_plot[i])
    testc = colors[i]
    scat3D._offsets3d = np.transpose(to_plot[i])
    scat3D.set_array(colors[i])
    return scat3D,


def init():
    scat3D.set_offsets([[],[],[]])
    plt.style.use('ggplot')
    plt.legend(handles=[red_patch, blue_patch])


ani = animation.FuncAnimation(fig, update_plot, init_func=init, blit=False, interval=100, frames=np.arange(total))

# ani.save("ani.gif", writer="imagemagick")

plt.show()