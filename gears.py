from math import sqrt, radians, cos, sin, tan, pi
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

teeth = np.array([28,30,34,38,43,50,60,75,100,150,300,400])
Y = np.array([0.353, 0.359, 0.371, 0.384, 0.397,0.409,0.422,
              0.435,0.447,0.46,0.472,0.48])
form_factor = CubicSpline(teeth, Y)


class Mesh():

    def __init__(self, g, p):
        p.rotate(g.n/p.n * g.w)

        #A = 36.755
        #B = 1.236
        #self.Kv = (1+sqrt(200 * self.v)/A)**B
        self.Kv = 1 + g.v/6.1

        bending_constant = g.Wt * self.Kv / M
        self.g_bending = bending_constant / (g.f * g.Y)
        self.p_bending = bending_constant / (p.f * p.Y)

        contact_constant = ZE * sqrt(g.Wt * self.Kv * (1/g.r + 1/p.r) / cos(PHI))
        self.g_contact = contact_constant / sqrt(g.f)
        self.p_contact = contact_constant / sqrt(p.f)

    def display(self, label):
        bsf = SY_GEAR * 1.5 / max([self.g_bending, self.p_bending])
        csf = (SC_GEAR * 1.5 / max([self.g_contact, self.p_contact]))**2
        print('{0:8}|{1:^8.1f}|{2:^8.1f}'.format(label, bsf, csf))


class Shaft():
    def __init__(self, g1, g2):
        g2.rotate(g1.w)

        self.dim1 = (25 + g2.f) // 5 * 5
        self.dim2 = 150 - (25 + g1.f) // 5 * 5
        self.forcepos = [9.5, self.dim1, self.dim2, 140.5]
        shoulders = [14, self.dim1 - g2.f/2, self.dim2 + g1.f/2, 136]
        self.d = np.array([14, 20, 20, 14])

        x = np.linspace(0, 150, 1501)
        d = np.where(x > shoulders[0], 20**4, 15**4)
        d = np.where(x > shoulders[1], 50**4, d)
        d = np.where(x > shoulders[2], 20**4, d)
        d = np.where(x > shoulders[3], 15**4, d)
        d = d * pi/64

        M = self.getMoments(g2.Wr, -g1.Wr, np.linspace(0, 150, 1501))
        slope = cumulative_trapezoid(M/d, dx=0.1, initial=0)
        deflection = cumulative_trapezoid(slope, dx=0.1, initial=0)
        f, a = plt.subplots(1,3)
        a[0].plot(deflection)
        a[1].plot(slope)
        a[2].plot(M/d)
        plt.show()

        Mz = self.getMoments(g2.Wr, -g1.Wr, shoulders)
        My = self.getMoments(-g2.Wt, -g1.Wt, shoulders)
        self.M = np.sqrt(My**2 + Mz**2)
        self.T = np.array([0, H, H, 0]) / (g1.w * pi/30)
    
    def getMoments(self, f1, f2, p):
        x = self.forcepos
        f0 = - (f1 * (x[1] - x[3]) + f2 * (x[2] - x[3])) / (x[0] - x[3])
        f3 = - (f0 + f1 + f2)
        f = [f0, f1, f2, f3]

        moments = []
        for pos in p:
            m = 0
            for i in range(len(x)):
                if pos < x[i]: break
                m += f[i] * (pos - x[i])
            moments.append(m / 1000)
        
        return np.array(moments)
    
    def display(self, label):

        ka = 3.04 * SUT_SHAFT**(-0.217)
        kb = 1.24 * self.d**(-0.107)
        Se = 0.5 * ka * kb * SUT_SHAFT
        Kf, Kfs = 1 + Q * (KT - 1), 1 + QS * (KTS - 1)

        fatigue_constant = 16 / (pi * (self.d)**3)
        sigma_a = fatigue_constant * 2 * Kf * self.M * 1000
        sigma_m = fatigue_constant * sqrt(3) * Kfs * self.T * 1000

        fsf = (sigma_a/Se + sigma_m/SUT_SHAFT)**(-1)
        ysf = SY_SHAFT/np.sqrt(sigma_a**2 + sigma_m**2)
        print('{0:8}|{1:^8.1f}|{2:^8.1f}'.format(label, np.min(fsf), np.min(ysf)))


class Gear():
    def __init__(self, n, f):
        self.n = n
        self.f = f

        self.d = M * self.n
        self.Y = form_factor(self.n)
        self.r = self.d * sin(PHI) / 2

    def rotate(self, w=20):
        self.w = w
        self.v = (self.w * pi/30) * (self.d/2000)
        self.Wt = H / self.v
        self.Wr = self.Wt * tan(PHI)

    def display(self, label):
        print('{0:8}|{1:^8d}|{2:^8.1f}|{3:^8.1f}'
              .format(label, self.n, self.w, self.d))


PHI = radians(20)
M = 1.5
H = 19.6 * 20 * pi/30

ZE = 191 
SY_GEAR, SC_GEAR = 200, 450
SC_GEAR = 450

SUT_SHAFT, SY_SHAFT = 1000, 900
KT, KTS = 2.7, 2.2
Q, QS = 0.92, 0.92


N2 = 45
N3 = 75
N4 = 80
N5 = round(N2*N4/(N3*1.175))

gear2 = Gear(N2, 16)
gear3 = Gear(N3, 8)
gear4 = Gear(N4, 8)
gear5 = Gear(N5, 8)

gear2.rotate()
mesh23 = Mesh(gear2, gear3)
shaft34 = Shaft(gear3, gear4)
mesh45 = Mesh(gear4, gear5)

print(' '*12 + 'N' + ' '*6 + 'omega' + ' '*6 + 'd')
gear2.display('Gear2')
gear3.display('Gear3')
gear4.display('Gear4')
gear5.display('Gear5')

print('\n' + ' '*11 + 'BSF' + ' '*6 + 'CSF')
mesh23.display('Mesh23')
mesh45.display('Mesh45')

print('\n' + ' '*11 + 'FSF' + ' '*6 + 'YSF')
shaft34.display('Shaft34')