import math
import matplotlib.pyplot as plt

EPS_START = 0.95
EPS_END = 0.01
EPS_DECAY = 300
steps_done = 0
eps_threshold_values = []

while steps_done < 1000:  # You can adjust the number of steps for the plot
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    eps_threshold_values.append(eps_threshold)
    steps_done += 1

plt.plot(range(len(eps_threshold_values)), eps_threshold_values)
plt.xlabel("Steps")
plt.ylabel("Epsilon Threshold")
plt.title("Epsilon Threshold Decay Over Steps")
plt.grid()
plt.show()
plt.savefig("./threshold.png")
