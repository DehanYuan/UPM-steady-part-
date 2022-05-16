import math

import numpy as np


np.seterr('raise')

def DiqtF(XC, YC, XS, YS, tau_x, tau_y, Q):
    # Number of panels
    numPan = len(XC)  # Number of panels/control points


    Diqt = np.zeros([numPan, Q])
    for i in range(numPan):
        for q in range(Q):
            rih = math.sqrt((XC[i]-XS[q])**2 + (YC[i]-YS[q])**2)
            Vihx = (YS[q] - YC[i]) / (2 * np.pi * rih**2)
            Vihy = (XC[i] - XS[q]) / (2 * np.pi * rih**2)
            Diqt[i,q] = Vihx * tau_x + Vihy * tau_y


    return Diqt

