import numpy as np
import math

import scipy.integrate as sci
import scipy.interpolate as scp

# Constantes
phi_max = 10 * math.pi           # Phi max, désactive le solveur d'intégration à partir d'un phi > 5 * pi (2 tour de trou noir + 1/2 tour)

class BlackHole:
    def __init__(self, Rs, D, FOV, Width, Height):
        self.Rs             = Rs
        self.D              = D
        self.FOV_X          = FOV
        self.FOV_Y          = FOV * (Height / Width)
        self.PPDX           = Width / self.FOV_X
        self.PPDY           = Height / self.FOV_Y
        self.AlphaFinder    = self.FOV_X / 2

        self.Integration(45)

        #self.SeenAngle, self.DeviatedAngle = self.Trajectories()
        #self.Interpolation                 = scp.interp1d(self.SeenAngle, self.DeviatedAngle, kind='linear', bounds_error=True)

    def GetValue(self):
        return self.SeenAngle, self.DeviatedAngle

    def DifferentialEquation(self, phi, u):
        """ Fonction représentant l'équation non linéaire suivante : (d²u(ɸ))/dɸ²= (3Rs)/2 * u² - u """
        v0 = u[1] # u[1] = u'
        v1 = ((3 * self.Rs) / 2) * (u[0] ** 2) - u[0] # u''
        print(f"u0'({phi}) = {v0} | u0''({phi}) = {v1}")
        return [v0, v1]

    def IsInsideBlackHole(self, phi, u):
        """ Le solveur va chercher pour quel valeur de phi, u[0] - Rs = 0, et arrêtera l'intégration si cela arrive """
        if u[0] - self.Rs == 0: # Gère l'exception de la division par 0
            return 0
        else:
            return (1/u[0]-self.Rs)
    IsInsideBlackHole.terminal = True

    def Integration(self, alpha):
        """ Intègre la fonction "DifferentialEquation" avec solveivp pour un angle alpha"""
        if alpha == 0: # Ici, le photon se dirige droit dans le trou noir, on retourne directement une distance r = 0 et un angle phi = 0
            return [0], [0]
        if alpha == 180: # Ici, le photon se dirige droit à l'opposé du trou noir, on se retrouve avec un angle phi = 0, et une distance r = D
            return [self.D], [0]
        y = [1/self.D, 1 / (self.D * math.tan(math.radians(alpha)))] # Condition initiale pour la position du photon et sa vitesse (u = 1/D; u' = 1 / ( D * tan(alpha) ))
        sol = sci.solve_ivp(fun=self.DifferentialEquation, t_span=[0, phi_max], y0=y, method="Radau", dense_output=False, events=[self.IsInsideBlackHole])
        for i in range(len(sol.t)):
            print(f"Phi = {sol.t[i]}, y = {sol.y[i]}");
        #phi = sol.t
        #r = abs(1/sol.y[0,:])
        #print(sol.t)
        #print(r)
        #return r, phi

    def SearchAlphaMin(self):
        """ Cherche le dernier angle alpha où le photon est pris dans le trou noir """
        alpha_min = float(0)
        last_alpha = int(0)
        for alpha in range(0, 180, 5):
            r, phi = self.Integration(alpha)
            if r[-1] > 1.1 * self.Rs:
                last_alpha = alpha
                break
        if last_alpha - 5 > 0:
            alpha_min = last_alpha - 5

        delta = 1 / self.PPDX
        for i in range(int(5/delta)):
            r, phi = self.Integration(alpha_min + i * delta)
            if r[-1] > 1.1 * self.Rs:
                alpha_min = alpha_min + i * delta
                break
        return alpha_min

    def Trajectories(self):
        seen_angle = []
        deviated_angle = []
        alpha_min = self.SearchAlphaMin()
        points = 40

        for i in range(6):
            for alpha in np.linspace(self.AlphaFinder, alpha_min, num = points):
                if alpha > 180:
                    break
                r, phi = self.Integration(alpha)
                if r[-1] > 1.01 * self.Rs:
                    seen_angle.append(180 - alpha)
                    deviated_angle.append(math.degrees( ( phi[-1] + math.asin( self.D / r[-1] * math.sin( phi[-1] ) ) ) ) )
            self.AlphaFinder = alpha_min + (self.AlphaFinder - alpha_min) / (points / 3 + 1)
            points = 10

        return seen_angle, deviated_angle
