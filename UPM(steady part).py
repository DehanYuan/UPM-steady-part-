import numpy as np
import math as math
import sympy as sym

from XFOIL import XFOIL
from IJ_SPM import IJ_SPM  #  To be used compute the normal and tangent component of induced velocity on airfoil panel from airfoil panel source
from KL_VPM import KL_VPM  #  To be used compute the normal and tangent component of induced velocity on airfoil panel from airfoil panel vortex
from Tit_Tin import Tit_Tin  # To be used compute the normal and tangent component of induced velocity on airfoil panel from TEV
from Ait_Ain import Ait_Ain  # To be used compute the normal and tangent component of induced velocity on airfoil panel from LSV
from DiqtF import DiqtF  # To be used compute the normal component of induced velocity on airfoil panel from separate vortices
from DiqnF import DiqnF  # To be used compute the tangent component of induced velocity on airfoil panel from separate vortices
from WihnF import WihnF  # To be used compute the normal component of induced velocity on airfoil panel from wake votrtices
from WihtF import WihtF  # To be used compute the tangent component of induced velocity on airfoil panel from wake votrtices

# Flag to specify creating or loading airfoil
flagAirfoil = [1,  # Create specified NACA airfoil in XFOIL
               0]  # Load Selig-format airfoil from directory

# User-defined knowns
Vinf = 1  # Freestream velocity [] (just leave this at 1)
AoA = 8  # Angle of attack [deg]
NACA = '0012'  # NACA airfoil to load [####]

# Initialize parameters
time_step = 0.02  # Setting time_step
ts = 1  # Setting the time you want to calculate

Kt = ts / time_step  # Compute the number of steps

# Convert angle of attack to radians
AoAR = AoA * (np.pi / 180)  # Angle of attack [rad]

# Plotting flags
flagPlot = [0,  # Airfoil with panel normal vectors
            0,  # Geometry boundary pts, control pts, first panel, second panel
            1,  # Cp vectors at airfoil surface panels
            1,  # Pressure coefficient comparison (XFOIL vs. VPM)
            0,  # Airfoil streamlines
            0]  # Pressure coefficient contour

# %% XFOIL - CREATE/LOAD AIRFOIL

# PPAR menu options
PPAR = ['101',  # "Number of panel nodes"
        '4',  # "Panel bunching paramter"
        '1',  # "TE/LE panel density ratios"
        '1',  # "Refined area/LE panel density ratio"
        '1 1',  # "Top side refined area x/c limits"
        '1 1']  # "Bottom side refined area x/c limits"

# Call XFOIL function to obtain the following:
# - Airfoil coordinates
# - Pressure coefficient along airfoil surface
# - Lift, drag, and moment coefficients
xFoilResults = XFOIL(NACA, PPAR, AoA, flagAirfoil)

# Separate out results from XFOIL function results
afName = xFoilResults[0]  # Airfoil name
xFoilX = xFoilResults[1]  # X-coordinate for Cp result
xFoilY = xFoilResults[2]  # Y-coordinate for Cp result
xFoilCP = xFoilResults[3]  # Pressure coefficient
XB = xFoilResults[4]  # Boundary point X-coordinate
YB = xFoilResults[5]  # Boundary point Y-coordinate
xFoilCL = xFoilResults[6]  # Lift coefficient
xFoilCD = xFoilResults[7]  # Drag coefficient
xFoilCM = xFoilResults[8]  # Moment coefficient

# Change the coordinate of XB and YB
t1 = XB
t2 = YB
for i in range(len(XB)):
    XB = t1 * np.cos(-AoAR) - t2 * np.sin(-AoAR)
    YB = t1 * np.sin(-AoAR) + t2 * np.cos(-AoAR)
# Number of boundary points and panels
numPts = len(XB)  # Number of boundary points
numPan = numPts - 1  # Number of panels (control points)

# %% CHECK PANEL DIRECTIONS - FLIP IF NECESSARY

# Check for direction of points
edge = np.zeros(numPan)  # Initialize edge value array
for i in range(numPan):  # Loop over all panels
    edge[i] = (XB[i + 1] - XB[i]) * (YB[i + 1] + YB[i])  # Compute edge values

sumEdge = np.sum(edge)  # Sum all edge values

# If panels are CCW, flip them (don't if CW)
if (sumEdge < 0):  # If panels are CCW
    XB = np.flipud(XB)  # Flip the X-data array
    YB = np.flipud(YB)  # Flip the Y-data array

# Initialize variables
XC = np.zeros(numPan)  # Initialize control point X-coordinate array
YC = np.zeros(numPan)  # Initialize control point Y-coordinate array
S = np.zeros(numPan)  # Initialize panel length array
phi = np.zeros(numPan)  # Initialize panel orientation angle array [deg]

# Find geometric quantities of the airfoil
for i in range(numPan):  # Loop over all panels
        XC[i] = 0.5 * (XB[i] + XB[i + 1])  # X-value of control point
        YC[i] = 0.5 * (YB[i] + YB[i + 1])  # Y-value of control point
        dx = XB[i + 1] - XB[i]  # Change in X between boundary points
        dy = YB[i + 1] - YB[i]  # Change in Y between boundary points
        S[i] = (dx ** 2 + dy ** 2) ** 0.5  # Length of the panel
        phi[i] = math.atan2(dy, dx)  # Angle of panel (positive X-axis to inside face)
        if (phi[i] < 0):  # Make all panel angles positive [rad]
            phi[i] = phi[i] + 2 * np.pi

# Compute angle of panel normal w.r.t. horizontal and include AoA
delta = phi + (np.pi / 2)  # Angle from positive X-axis to outward normal vector [rad]
beta = delta - AoAR  # Angle between freestream vector and outward normal vector [rad]
beta[beta > 2 * np.pi] = beta[beta > 2 * np.pi] - 2 * np.pi  # Make all panel angles between 0 and 2pi [rad]

# Separation vortex and Trailing-edge vortex parameters definition
DeltaS_T = 0.1  # Initialize DeltaS_T
Theta_T = 15 / np.pi  # Initialize Theta_T
DeltaS_Sl = 0.1    # Initialize DeltaS_S
Theta_Sl = 15 / np.pi  # Initialize Theta_S

# Parameters initialize
Q = 0  # number of Separate vortex
M = 0  # number of Wake vortex

gammal = 0  # gamma value in previous situation
XSlt = np.zeros(Q)
YSlt = np.zeros(Q)
V_itc = np.zeros(numPan)  # source strength of airfoil (different from panel to panel)
ABM = np.zeros(numPan)
BBM = np.zeros([numPan, numPan])
CBM = np.zeros(numPan)
  # BC parameters

ACM = np.zeros(numPan)
BCM1 = np.zeros([numPan, numPan])
BCM = np.zeros([numPan, numPan])
CCM = np.zeros(numPan)
DCM = np.zeros(numPan)
ECM = np.zeros(numPan)
FCM = np.zeros(numPan)
GCM = np.zeros(numPan)
HCM = np.zeros(numPan)

l = sum(S)  # compute the perimeter of airfoil
Vs_x = np.zeros(numPan)
Vs_y = np.zeros(numPan)
for i in range(numPan):
    Vs_x[i] = Vinf
    Vs_y[i] = 0

XW = np.zeros(M)
YW = np.zeros(M)
XS = np.zeros(Q)
YS = np.zeros(Q)
Gamma_wh = np.zeros(M)
Gamma_Sq = np.zeros(Q)
xSl = np.zeros(int(Kt))
ySl = np.zeros(int(Kt))  # Wake vortex location, Separation vortex location, LSV and TEV define

tau_x = np.zeros(numPan)
tau_y = np.zeros(numPan)
delta_i = np.zeros(numPan)  # normal vector angle of airfoil panels

Cp = np.zeros(numPan)
Phi = np.zeros(numPan)  # Pressure parameters
gamma = sym.symbols('gamma')  # vortex strength of airfoil
Phil = np.zeros(numPan)
# CL = np.zeros(int(Kt))
# CM = np.zeros(int(Kt))  # Coefficients definetion
# Determine fixed separation location
nfxsl = 0
fxsl = np.zeros(numPan)
for i in range(numPan):
    fxsl[i] = XB[i] - nfxsl
islindex = np.argmin(fxsl)
x_isl = XB[islindex]
y_isl = YB[islindex]  # Define latest separation vortex associate with the airfoil location

Xle = min(XB)
Yle = min(YB)
YCP = Yle
ile = np.argmin(XB)
numPlee = 100
XBP = np.zeros(numPlee)
XCP = np.zeros(numPlee)
XBP[0] = Xle
ki = np.zeros(numPan)  # define the slope of each airfoil panel
bi = np.zeros(numPan)  # define intercept of each airfoil panel
dSli = np.zeros(islindex)
dSui = np.zeros(numPan)
countl = 0
countu = 0

for i in range(numPan):
    delta_i[i] = phi[i] + np.pi / 2
    tau_x[i] = (XB[i + 1] - XB[i]) / S[i]  # cos(panel incline angle)
    tau_y[i] = (YB[i + 1] - YB[i]) / S[i]  # sin(panel incline angle)


xT = XB[0] + 0.5 * DeltaS_T * np.cos(Theta_T)
yT = YB[0] + 0.5 * DeltaS_T * np.sin(Theta_T)  # Compute Trailing edge vortex location
xTB = XB[0] + DeltaS_T * np.cos(Theta_T)
yTB = YB[0] + DeltaS_T * np.sin(Theta_T)  # Compute Trailing edge vortex boundary point location

xSl = x_isl + 0.5 * DeltaS_Sl * np.cos(Theta_Sl)
ySl = y_isl + 0.5 * DeltaS_Sl * np.sin(Theta_Sl)  # Define latest separation vortex control point location

# unit TEV strength computation
gamma_T = (gamma - gammal) * l / DeltaS_T - gamma * DeltaS_Sl

# Unit LSV strength computation
gamma_Sl = gamma

# Geometric integrals for SPM and VPM (normal [I,K] and tangential [J,L])
I, J = IJ_SPM(XC, YC, XB, YB, phi, S)
K, L = KL_VPM(XC, YC, XB, YB, phi, S)
Wihn = WihnF(XC, YC, XW, YW, delta_i, M)
Tin, Tit = Tit_Tin(XC, YC, xT, yT, phi, DeltaS_T, Theta_T)
Ain, Ait = Ait_Ain(XC, YC, x_isl, y_isl, phi, DeltaS_Sl, Theta_Sl)
Diqn = DiqnF(XC, YC, XS, YS, delta_i, Q)
Wiht = WihtF(XC, YC, XW, YW, tau_x, tau_y, M)
Diqt = DiqtF(XC, YC, XS, YS, tau_x, tau_y, Q)
# Compute normal velocity of airfoil panels
for i in range(numPan):
    ABM[i] = 2 * np.pi * (Vs_x[i] * np.cos(delta_i[i]) + Vs_y[i] * np.sin(delta_i[i]))  # Compute AM term of BC.eq.

for i in range(numPan):
    for j in range(numPan):
        if (i==j):
            BBM[i, j] = np.pi
        else:
            BBM[i, j] = I[i, j]  # Compute BM term of BC.eq.

for i in range(numPan):
        CBM[i] = -sum(K[i])   # Compute CM term of BC.eq.

DBM = np.dot(Wihn, Gamma_wh)  # Compute DM term of BC


BBM_inv = np.linalg.inv(BBM)
lamb = np.dot(BBM_inv,  - ABM - CBM * gamma)  # Obtain source strength in terms of unit vortex strength

# Compute tangent velocity of airfoil panel
for i in range(numPan):
    ACM[i] = (Vs_x[i] * tau_x[i] + Vs_y[i] * tau_y[i])

BCM = np.dot(J, lamb) / (2 * np.pi)   # Compute B term of tangent velocity to calculate KC
for i in range(numPan):
    CCM[i] = sum(L[i]) / (2 * np.pi)    # Compute C term of tangent velocity to calculate KC
CCM = -CCM * gamma
DCM = np.dot(Wiht, Gamma_wh)  # Compute D term of tangent velocity to calculate KC

GCM = np.dot(Diqt, Gamma_Sq)  # Compute G term of tangent velocity to calculate KC
for i in range(numPan):
    HCM[i] = 0.5
HCM = HCM * gamma
V_it = ACM + BCM + CCM + HCM  # Compute the tangent velocity of airfoil panel

sym.init_printing()
Kutta = sym.Eq(V_it[numPan - 1], - V_it[0])
ans = sym.solve(Kutta, gamma)
gammav = ans[0]
lamb = np.dot(BBM_inv, -ABM - CBM * gammav)

for i in range(numPan):
    ACM[i] = (Vs_x[i] * tau_x[i] + Vs_y[i] * tau_y[i])

BCM = np.dot(J, lamb) / (2 * np.pi)   # Compute B term of tangent velocity to calculate KC
for i in range(numPan):
    CCM[i] = sum(L[i]) / (2 * np.pi)    # Compute C term of tangent velocity to calculate KC
CCM = -CCM * gammav
DCM = np.dot(Wiht, Gamma_wh)  # Compute D term of tangent velocity to calculate KC

GCM = np.dot(Diqt, Gamma_Sq)  # Compute G term of tangent velocity to calculate KC
for i in range(numPan):
    HCM[i] = 0.5
HCM = HCM * gammav
V_itv = ACM + BCM + CCM + HCM  # Compute the tangent velocity of airfoil panel


for i in range(numPan):
    Cp[i] = 1 - (V_itv[i] / Vinf) ** 2  # Compute pressure coefficient on panel i
CL = sum(-Cp * S * np.sin(delta_i))  # Lift force coefficient []
print("  UPM : %2.8f" % CL)  # From this SPVP code
print("  XFOIL: %2.8f" % xFoilCL)  # From XFOIL program














