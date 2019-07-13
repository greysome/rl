import numpy as np
from time import sleep
import matplotlib.pyplot as plt

class LivePlotter(object):
    def __init__(self):
        plt.ion()
        self.vals = []
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.line, = self.ax.plot(self.vals, 'r-')

    def update(self, val):
        self.vals.append(val)
        self.ax.set_xlim(0, len(self.vals))
        self.ax.set_ylim(0, max(self.vals)+min(self.vals))
        self.line.set_xdata(range(len(self.vals)))
        self.line.set_ydata(self.vals)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        plt.close()
