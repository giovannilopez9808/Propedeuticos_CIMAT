from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
X = np.arange(-2.5, 2.5, 0.25)
Y = np.arange(-2.5, 2.5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = (1-X**2)+100*(Y-X**2)**2

fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.view_init(15, 30)
ax.plot_surface(X, Y, Z,
                rstride=1,
                cstride=1,
                cmap="inferno")
plt.savefig("Rosenbrock.png")
