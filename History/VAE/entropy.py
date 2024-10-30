import matplotlib.pyplot as plt
import numpy as np

# distribution
x1 = np.linspace(-3,3,100)
y1 = np.abs(np.cos(x1))

# appoximation
y2 = np.abs(np.sin(x1-1))

# entropy
xentropy1 = y1
yentropy1 = np.log2(1/y1)

# approx entropy
xentropy2 = y2

print(y1,sum(np.abs(x1)))

fig, axes = plt.subplots(2,2)

l1 = axes[0][0].plot(x1,y1)
l2 = axes[0][1].plot(x1,y2)

l3 = axes[1][0].plot(xentropy1,yentropy1)
l4 = axes[1][1].plot(xentropy2,yentropy1)


axes[0][0].set_title("Real P(x)")
axes[0][1].set_title("Estimation Q(x)")
axes[1][0].set_title("Entropy w.r.t P(x)")
axes[1][1].set_title("Entropy w.r.t  Q(x)")

plt.show()