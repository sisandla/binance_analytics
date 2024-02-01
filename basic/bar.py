"""
==============
bar(x, height)
==============

See `~matplotlib.axes.Axes.bar`.
"""
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

plt.style.use('_mpl-gallery')

# make data:
x = 1 + np.arange(8)
y = [4, 5, 3, 4, 6, 6, 2, 3]

def collatz(num):
       loop = 0
       step = 0
       series = []

       while(loop != 1):
              if (num%2 == 0):
                     num = int(num/2)
                     series.append(num)

              elif(num%2 != 0):
                     num = int(3*num + 1)
                     series.append(num)

              elif(num == 1):
                     series.append('1')

              loop = num
              step = step + 1

       return series, step

cy, step = collatz(20)
cx = 1 + np.arange(step)


print(list(cx))
print(x)

# plot
fig, ax = plt.subplots()

ax.bar(cx, cy, width=1, edgecolor="white", linewidth=0.7)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

plt.show()
