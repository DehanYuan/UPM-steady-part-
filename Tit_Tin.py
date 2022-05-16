import numpy as np
import math as math

np.seterr('raise')


def Tit_Tin(XC, YC, xTB, yTB, phi, DeltaS_T, Theta_T):
    # Number of panels
    numPan = len(XC)  # Number of panels

    # Initialize arrays
    Tin = np.zeros(numPan)
    Tit = np.zeros(numPan)

    # Compute integral
    for i in range(numPan):
        A = -(XC[i] - xTB) * np.cos(Theta_T) - (YC[i] - yTB) * np.sin(Theta_T)  # A term
        B = (XC[i] - xTB) ** 2 + (YC[i] - yTB) ** 2  # B term
        Cn = -np.cos(phi[i] - Theta_T)  # C term (normal)
        Dn = (XC[i] - xTB) * np.cos(phi[i]) + (YC[i] - yTB) * np.sin(phi[i])  # D term (normal)
        Ct = np.sin(Theta_T - phi[i])  # C term (tangential)
        Dt = (XC[i] - xTB) * np.sin(phi[i]) - (YC[i] - yTB) * np.cos(phi[i])  # D term (tangential)
        E = np.sqrt(B - A ** 2)  # E term
        if (E == 0 or np.iscomplex(E) or np.isnan(E) or np.isinf(
                E)):
            Tin[i] = 0
            Tit[i] = 0
        else:

            term1 = 0.5 * Cn * np.log((DeltaS_T ** 2 + 2 * A * DeltaS_T + B) / B)  # First term in K equation
            term2 = ((Dn - A * Cn) / E) * (
                    math.atan2((DeltaS_T + A), E) - math.atan2(A, E))  # Second term in K equation
            Tin[i] = term1 + term2


            term1 = 0.5 * Ct * np.log((DeltaS_T ** 2 + 2 * A * DeltaS_T + B) / B)  # First term in L equation
            term2 = ((Dt - A * Ct) / E) * (
                    math.atan2((DeltaS_T + A), E) - math.atan2(A, E))  # Second term in L equation
            Tit[i] = term1 + term2


            # Zero out any problem values
            if (np.iscomplex(Tin[i]) or np.isnan(Tin[i]) or np.isinf(
                    Tin[i])):  # If K term is complex or a NAN or an INF
                Tin[i] = 0  # Set K value equal to zero
            if (np.iscomplex(Tit[i]) or np.isnan(Tit[i]) or np.isinf(
                    Tit[i])):  # If L term is complex or a NAN or an INF
                Tit[i] = 0  # Set L value equal to zero

    return Tin, Tit  # Return both K and L matrices
