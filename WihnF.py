import math

import numpy as np


np.seterr('raise')

def WihnF(XC, YC, XW, YW, delta_i, M):
    # Number of panels
    numPan = len(XC)  # Number of panels/control points


    Wihn = np.zeros([numPan, M])
    for i in range(numPan):
        for h in range(M):
            rih = math.sqrt((XC[i]-XW[h])**2 + (YC[i]-YW[h])**2)
            Vihx = (YW[h] - YC[i]) / (2 * np.pi * rih**2)
            Vihy = (XC[i] - XW[h]) / (2 * np.pi * rih**2)
            Wihn[i, h] = Vihx * np.cos(delta_i) + Vihy * np.sin(delta_i)


    return Wihn

