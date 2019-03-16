import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Computer Modern'

ista 	= pd.read_csv('output/ISTA.csv')
fista 	= pd.read_csv('output/FISTA.csv')

k = ista['step'].values 

plt.plot(k, ista['objective'].values, label='ISTA', color='red')
plt.plot(k, fista['objective'].values, label='FISTA', color='blue')

plt.xlabel('t')

plt.legend()
plt.savefig('output/convergence.png',bbox_inches='tight',pad_inches=0,
	dpi=100,figsize=(1,1))