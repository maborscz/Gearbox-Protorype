from math import sqrt, radians, cos, sin, tan, pi
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.integrate import cumulative_trapezoid


# Data on form factor (Shigley Table 14-2)
teeth = np.array([28,30,34,38,43,50,60,75,100,150,300,400])
Y = np.array([0.353, 0.359, 0.371, 0.384, 0.397,0.409,0.422,
              0.435,0.447,0.46,0.472,0.48])
form_factor = CubicSpline(teeth, Y) # For interpolation


# Object for defining a mesh between two gears
class Mesh():

    '''
    IMPORTANT CALCULATIONS
    '''
    def __init__(self, g1, g2):
        g2.rotate(g1.n/g2.n * g1.w) # Rotate second gear according to gear ratio

        self.Kv = 1 + g1.v/6.1 # Dynamic factor for cut gear profile (Shigley Eq 14-6b)

        # Bending stress (Shigley Eq 14-8)
        # sigma_b = K_v * W_t / (F * m * Y)
        bending_constant = g1.Wt * self.Kv / M
        self.g1_bending = bending_constant / (g1.f * g1.Y)
        self.g2_bending = bending_constant / (g2.f * g2.Y)

        # Contact stress (Shigley Eq 14-14)
        # |sigma_c| = ZE * [K_v * W_t / (F * cos(phi)) * (1/r1 + 1/r2)]^(1/2)
        contact_constant = ZE * sqrt(g1.Wt * self.Kv * (1/g1.r + 1/g2.r) / cos(PHI))
        self.g1_contact = contact_constant / sqrt(g1.f)
        self.g2_contact = contact_constant / sqrt(g2.f)

    '''
    IMPORTANT CALCULATIONS
    '''
    # Display minimum bending and contact safety factors (BSF, CSF)
    def display(self, label):

        # See Shigley Eq 14-41 and Eq 14-43
        Y_N, Z_N = 1.5, 1.5 # Stress cycle factors at 10^4 load cycles
        # Temperature factor = 1
        # Reliability factor = 1 (99 percent)
        # Hardness-ratio factor = 1
        bsf = SY_GEAR * Y_N / max([self.g1_bending, self.g2_bending]) # Choose minimum BSF
        csf = (SC_GEAR * Z_N / max([self.g1_contact, self.g2_contact]))**2 # Choose minimum CSF

        print('{0:8}|{1:^8.1f}|{2:^8.1f}'.format(label, bsf, csf))


# Object for defining a shaft connecting two gears
class Shaft():
    
    '''
    IMPORTANT CALCULATIONS
    '''
    def __init__(self, g1, g2):
        g2.rotate(g1.w) # Rotate second gear with same speed as first gear

        # x=0 is taken as the side near Gear 4
        dim1 = 28 # Student dimension near Gear 4
        dim2 = 122 # Student dimension near Gear 3

        # Locations of applied forces [bearing, gear, gear, bearing]
        self.fpos = np.array([9.5, dim1-g2.f/2, dim2+g1.f/2, 140.5])
        # Locations of shoulders including start and end faces
        self.sh = np.array([0, 14, dim1, dim2, 136, 150])

        self.x = np.linspace(0, 150, 1501) # Array of x-values to calculate data on
        # Diameter of shaft at each x-value
        self.diam = self.populate(self.sh[1:-1], [15, 20, 50, 20, 15])
        # Second moment of inertia (times elastic modulus) at each x-value
        # Second moment of inertia of a circle is pi*d^4/64
        self.EI = self.diam**4 * pi/64 * E_SHAFT * 10**(-3) 

        # Get shear, moment, slope and deflection in xy and xz plane
        self.dz = self.getData(-g2.Wt, -g1.Wt)
        self.dy = self.getData(g2.Wr, -g1.Wr)

        # Calculate total moment, slope and deflection (L2 norm)
        self.m = np.sqrt(self.dz[1]**2 + self.dy[1]**2)
        self.v = np.sqrt(self.dz[2]**2 + self.dy[2]**2)
        self.d = np.sqrt(self.dz[3]**2 + self.dy[3]**2)

        # Calculate torque, which is only present between the two gears 
        tau = H / (g1.w * pi/30)
        self.t = self.populate(self.fpos[1:-1], [0, tau, 0])

    '''
    IMPORTANT CALCULATIONS
    '''
    # Function for calulating shear, moment, slope and deflection
    # given forces at gears
    def getData(self, f1, f2):
        x0, x1, x2, x3 = self.fpos # Posititons of forces
        f0 = - (f1 * (x1 - x3) + f2 * (x2 - x3)) / (x0 - x3) # Force at bearing 1
        
        # Shear at each x-value
        v = self.populate(self.fpos, [0, f0, f0+f1, f0+f1+f2, 0])

        # Moment at each x-value, obtained by integrating shear
        m = cumulative_trapezoid(v, dx=0.0001, initial=0) # dx = 0.1 mm
        # Slope at each x-value, obtained by integrating moment/EI
        s = cumulative_trapezoid(m/self.EI, dx=0.0001, initial=0)
        # Deflection at each x-value, obtained by integrating slope
        d = cumulative_trapezoid(s, dx=0.0001, initial=0)

        # Calculate integration constants for slope and deflection,
        # given that the deflection at the bearing positions is zero
        A = - (d[int(x0*10)] - d[int(x3*10)]) / (x0 - x3)
        B = (d[int(x0*10)] * x3 - d[int(x3*10)] * x0) / (x0 - x3)

        s += A * 1000 # Add A to slope
        d += A*self.x + B # Add Ax+B to deflection

        return [v, m, s, d]
    
    # Display fatigue and yielding safety factors (FSF, YSF) and critical speed
    def display(self, label):
        fsf, ysf = self.getSF()
        omega = self.getCritSpeed()
        print('{0:8}|{1:^8.1f}|{2:^8.1f}|{3:^8d}'.format(label, fsf, ysf, omega))

    # Function used for creaing shear array and diameter array
    # Populates x with each value in val, changin value at each position in pos
    def populate(self, pos, val):
        pos = np.append(pos, self.x[-1] + 1)
        j = 0
        out = np.zeros_like(self.x)

        for i, x in enumerate(self.x):
            if x < pos[j]:
                v = val[j]
            else:
                v = (val[j] + val[j+1])/2
                j += 1
            out[i] = v

        return out
    
    '''
    IMPORTANT CALCULATIONS
    '''
    # Calulate fatigue and yielding safety factor
    def getSF(self):
        idx = (self.sh[1:-1] * 10).astype(int) # Position of critical points

        # Completely reversed simple loading
        Se_prime = 0.5 * SUT_SHAFT # Endurance limit (Shigley Eq 6-10)
        ka = 3.04 * SUT_SHAFT**(-0.217) # Surface factor for machined steel (Shigley Eq. 6-18)
        kb = 1.24 * self.diam[idx]**(-0.107) # Size factor (Shigley Eq. 6-19)
        # Loading factor kc = 1
        # Temperature factor kd = 1
        ke = 0.814 # Reliability factor at 99% (Shigley Table 6-4)
        Se = ka * kb * ke * Se_prime # Shigley Eq. 6-17
        Kf, Kfs = 1 + Q * (KT - 1), 1 + QS * (KTS - 1) # Shigley Eq. 6-32

        # Goodman distortion energy failure theory (Shigley Eq. 7-4, 7-5)
        # Mean bending stress is zero, alternating torque is zero
        fatigue_constant = 16 / (pi * (self.diam[idx])**3)
        sigma_a = fatigue_constant * 2 * Kf * self.m[idx] * 1000 # factor of 1000 used to convert to MPa
        sigma_m = fatigue_constant * sqrt(3) * Kfs * self.t[idx] * 1000 # factor of 1000 used to convert to MPa

        # Goodman line for failure (Shigley Eq. 6-41)
        fsf = (sigma_a/Se + sigma_m/SUT_SHAFT)**(-1)
        # Yielding safety factor (Shigley Eq. 7-15, 7-16)
        ysf = SY_SHAFT/np.sqrt(sigma_a**2 + sigma_m**2)

        return [np.min(fsf), np.min(ysf)] # Get minimum of safety factors at critical points

    '''
    IMPORTANT CALCULATIONS
    '''
    def getCritSpeed(self):
        self.c = (self.sh[1:] + self.sh[:-1]) / 2 # Position of centroids
        c = (self.c * 10).astype(int)
        l = self.sh[1:] - self.sh[:-1] # Lengths of lumped elements

        # Rayleigh's method for lumped masses (Shigley Eq. 7-23)
        sum1 = np.sum(self.diam[c]**2 * l * self.d[c])
        sum2 = np.sum(self.diam[c]**2 * l * self.d[c]**2)

        return int(sqrt(9.81 * sum1/sum2) * 30/pi) # Convert to rpm


'''
IMPORTANT CALCULATIONS
'''
class Gear():
    def __init__(self, n, f):
        self.n = n # Number of gear teeth
        self.f = f # Face width

        self.d = M * self.n # Diameter of gear
        self.Y = form_factor(self.n) # Form factor
        self.r = self.d * sin(PHI) / 2 # Radii of curvature of tooth profile (Shigley Eq. 14-12)

    def rotate(self, w=20):
        self.w = w # Rotate gear with angular velocity w
        self.v = (self.w * pi/30) * (self.d/2000) # Calculate speed of gear teeth
        self.Wt = H / self.v # Calculate tangential force at gear teeth
        self.Wr = self.Wt * tan(PHI) # Calulate radial force at gear teeth

    # Display number of gear teeth, angular velocity and gear diameter
    def display(self, label):
        print('{0:8}|{1:^8d}|{2:^8.1f}|{3:^8.1f}'
              .format(label, self.n, self.w, self.d))

'''
RAW DATA
'''
PHI = radians(20) # Pressure angle
M = 1.5 # Module
H = 19.6 * 20 * pi/30 # Power transmitted (torque x angular velocity)

ZE = 191 # Elastic coefficient of steel (Shigley Table 14-6)
SY_GEAR = 250 # Yield stress of mild steel (Bluescope HA250)
# Brinell hardness assumed to be 130 (Shigley Table A-20)
# Contact stress was found to be 2.22x130+200=480 MPa (Shigley Fig 14-5)
SC_GEAR = 480 

# https://www.flamehardening.com.au/wp-content/uploads/2016/05/EN36A.pdf
SUT_SHAFT, SY_SHAFT = 800, 650
E_SHAFT = 200

KT, KTS = 2.7, 2.2 # Stress concentration factors for sharp shoulder (Shigley Table 7-1)
Q, QS = 0.88, 0.9 # Notch sensitivites (Shigley Figure 6-26 and 6-27)

# Gear teeth numbers
N2 = 45
N3 = 75
N4 = 80
N5 = round(N2*N4/(N3*1.175)) # 1.175 is desired gear ratio

# Create Gears
gear2 = Gear(N2, 30) # Input gear has a ~30mm face width
gear3 = Gear(N3, 8) # All other gears have a 8mm face width
gear4 = Gear(N4, 8)
gear5 = Gear(N5, 8)

gear2.rotate() # Rotate G2
mesh23 = Mesh(gear2, gear3) # Mesh G2 and G3
shaft34 = Shaft(gear3, gear4) # Place G3 and G4 on shaft
mesh45 = Mesh(gear4, gear5) # Mesh G4 and G5

# Display results
print(' '*12 + 'N' + ' '*8 + 'w' + ' '*8 + 'd')
gear2.display('Gear2')
gear3.display('Gear3')
gear4.display('Gear4')
gear5.display('Gear5')

print('\n' + ' '*11 + 'BSF' + ' '*6 + 'CSF')
mesh23.display('Mesh23')
mesh45.display('Mesh45')

print('\n' + ' '*11 + 'FSF' + ' '*6 + 'YSF' +' '*6 + 'w_c')
shaft34.display('Shaft34')
