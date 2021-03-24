# -*- coding: utf-8 -*-
"""
This file makes projections of numbers of corona virus deaths based on an
exponential grow; 
A semilog figure is generated
This is a file for March 31 2020.
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.special as spp
doubling = 3 #this is the doubling time
death0 = 200. # these are the number of deaths on day 0 in our model
def deaths(day,dt,dd0):
    """ this calculates the deaths: dt=doubling time; dd0=deaths of day 0""" 
    return dd0*2**(day/dt)

days = np.linspace(-30,30,41)
ndeaths = deaths(days,doubling,death0 )

figure=plt.semilogy(days,ndeaths,".", color="k")

plt.xlabel("Day", fontname="Times New Roman",fontsize=16)
plt.ylabel("cumulative deaths (#individual)", fontname="Times New Roman",fontsize=16)
plt.title("Predicted deaths in US: doubling=3 days", fontname="Times New Roman",
          fontsize=20)

plt.grid()
plt.savefig("coronavrus.pdf")
plt.show()