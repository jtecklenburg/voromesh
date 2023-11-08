# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 14:28:49 2023

@author: jante
"""

import numpy as np


# def streamline(x):
#     z = x/1000
#     return 200*np.tanh(z)+1.0


def streamline(x):
    return 500*np.cos(x/2000)+1.0


def xymesh(xmin=-6000, xmax=6000, dx=20, dy=2, layer=39):

    x = list()
    y = list()

    x0 = xmin
    y0 = streamline(x0)

    x.append(x0)
    y.append(y0)

    while x0 < xmax:

        x1_guess = x0 + dx
        y1_guess = streamline(x1_guess)

        d_guess = dx - np.sqrt((x0-x1_guess)**2 + (y0-y1_guess)**2)

        x1 = x1_guess - d_guess/dx * (x1_guess-x0)
        y1 = streamline(x1)

        x.append(x1)
        y.append(y1)

        x0 = x1
        y0 = y1

    x = np.array(x)
    y = np.array(y)

    xx = x.copy()
    yy = y.copy()

    for j in range(layer):
        xneu = list()
        yneu = list()

        xneu.append(x[0])
        yneu.append(y[0]+dy)

        for i in range(len(x)-1):

            n1 = x[i+1]-x[i]
            n2 = y[i+1]-y[i]

            d_guess = np.sqrt(n1**2 + n2**2)

            xneu.append(x[i+1] - dy/d_guess * n2)
            yneu.append(y[i+1] + dy/d_guess * n1)

        xx = np.hstack((xx, xneu))
        yy = np.hstack((yy, yneu))

        x = xneu
        y = yneu

    return np.column_stack((xx, yy))
