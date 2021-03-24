# -*- coding: utf-8 -*-
"""
This file compares the binomial distribution to the equivalent Gaussian
 distribution for the probability that the number of 1000 democrats polled
 will be Elizabeth Warren supporters

This is a file for the minicourse for february 4 2020.
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.special as spp
a=spp.binom(1000,140)
n=1000.
p=0.14
def binomdist(number, frequency, probability):
    """probability of frequency events from sample of number""" 
    return spp.binom(number,frequency)*probability**frequency*(1-probability)**(number-frequency)
def gaussn(number,freq,probability):
    """Gaussian approx. to binomial"""
    mu=number*probability
    sigmasquared=number*probability*(1-probability)
    norm=1/(2.0*np.pi*sigmasquared)**(1/2)
    return norm*np.exp(-(freq-mu)**2/(2*sigmasquared))

x=np.linspace(100,200,101)

y=binomdist(n,x,p)
z=gaussn(n,x,p)
plt.plot(x,z, label="Gaussian", color="k")
plt.plot(x,y,".", label="binomial")
plt.xlabel("frequency (number of dems)", fontname="Times New Roman",fontsize=16)
plt.ylabel("probability", fontname="Times New Roman",fontsize=16)
plt.title("Comparison of binomial and Gaussian distributions", fontname="Times New Roman",fontsize=18)
plt.legend()
plt.grid()
plt.savefig("gaussvbin.pdf")
plt.show()



x10=np.linspace(100,200,11)
print(x10)
y10=gaussn(n,x10,0.14)
print(y10)