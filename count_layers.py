import numpy as np
d_p = []
with open("layer.txt", "r") as f:
    data = f.readlines()
    for d in data:
        d_p.append(int(d.strip()))
    print(np.mean(d_p))

u = 500
theta = 500
print(np.exp(-((400-u)**2)/(2*theta)))