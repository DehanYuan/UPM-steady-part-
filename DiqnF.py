import math

import numpy as np


np.seterr('raise')

def DiqnF(XC, YC, XS, YS, delta_i, Q):
    # Number of panels
    numPan = len(XC)  # Number of panels/control points


    Diqn = np.zeros([numPan, Q])
    for i in range(numPan):
        for q in range(Q):
            rih = math.sqrt((XC[i]-XS[q])**2 + (YC[i]-YS[q])**2)
            Viqx = (YS[q] - YC[i]) / (2 * np.pi * rih**2)
            Viqy = (XC[i] - XS[q]) / (2 * np.pi * rih**2)
            Diqn[i, q] = Viqx * np.cos(delta_i) + Viqy * np.sin(delta_i)


    return Diqn

