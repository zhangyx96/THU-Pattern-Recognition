import matplotlib.pyplot as plt

a = [0.8725,0.881,0.7415,0.887,0.885,0.901,0.918,0.9,0.88,0.8245]
b = ['airplane/frog','airplane/horse','airplane/ship','airplane/truck',
     'frog/horse','frog/ship','frog/truck','horse/ship','horse/truck','ship/truck']
plt.barh(range(len(a)),a,tick_label = b)
plt.show()