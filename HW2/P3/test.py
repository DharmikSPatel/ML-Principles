import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(xlim=(-10,10), ylim=(-10, 10))

x = np.linspace(-1, 1)
y = x**2
ax.plot(x,y)

# Show the figure
plt.show()