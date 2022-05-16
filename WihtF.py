import math

import numpy as np


np.seterr('raise')

def WihtF(XC, YC, XW, YW, tau_x, tau_y, M):
    # Number of panels
    numPan = len(XC)  # Number of panels/control points


    Wiht = np.zeros([numPan, M])
    for i in range(numPan):
        for h in range(M):
            rih = math.sqrt((XC[i]-XW[h])**2 + (YC[i]-YW[h])**2)
            Vihx = (YW[h] - YC[i]) / (2 * np.pi * rih**2)
            Vihy = (XC[i] - XW[h]) / (2 * np.pi * rih**2)
            Wiht[i, h] = Vihx * tau_x + Vihy * tau_y


    return Wiht

