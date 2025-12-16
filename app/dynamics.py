# dynamics.py
# Simulación del Quadrotor: Ecuaciones de Movimiento

import numpy as np
from scipy.integrate import solve_ivp

class QuadrotorDynamics:
    """
    Modelado de la dinámica no lineal de un Quadrotor (12 estados).
    """
    def __init__(self):
        # --- Parámetros Físicos ---
        self.m = 0.5    # Masa (kg)
        self.g = 9.81   # Gravedad (m/s^2)
        self.L = 0.25   # Distancia del centro al rotor (m)
        self.Ixx = 4.856e-3  # Momento de inercia x (kg*m^2)
        self.Iyy = 4.856e-3  # Momento de inercia y
        self.Izz = 8.801e-3  # Momento de inercia z
        self.I = np.array([self.Ixx, self.Iyy, self.Izz])
        self.kf = 1.0  # Coeficiente de fuerza/empuje (simplificado)
        self.km = 0.05 # Coeficiente de momento/arrastre
        
    def _state_derivative(self, t, X, U):
        """
        Calcula el vector de derivadas de estado (dX/dt).
        X = [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r] (12 estados)
        U = [F1, F2, F3, F4] (Fuerzas/Empuje de los 4 rotores)
        """
        
        # 1. Decodificar estados
        # Posición y Velocidad Lineal
        x, y, z = X[0], X[1], X[2]
        vx, vy, vz = X[3], X[4], X[5]
        
        # Actitud (Ángulos de Euler)
        phi, theta, psi = X[6], X[7], X[8]
        
        # Velocidad Angular (p, q, r)
        p, q, r = X[9], X[10], X[11]

        # 2. Variables de Control (Empujes individuales y agregados)
        F1, F2, F3, F4 = U[0], U[1], U[2], U[3]
        
        # Empuje total hacia arriba (en el cuerpo)
        T = F1 + F2 + F3 + F4
        
        # Torques (Tau)
        Tau_phi = self.L * (F1 - F3)      # Roll (alrededor de x)
        Tau_theta = self.L * (F2 - F4)    # Pitch (alrededor de y)
        Tau_psi = self.km * (F1 - F2 + F3 - F4) # Yaw (alrededor de z)
        Tau = np.array([Tau_phi, Tau_theta, Tau_psi])

        # 3. Dinámica Traslacional (Velocidad lineal)
        # Matriz de Rotación de Cuerpo (B) a Mundo (W)
        R_BW = np.array([
            [np.cos(psi)*np.cos(theta), np.cos(psi)*np.sin(theta)*np.sin(phi) - np.sin(psi)*np.cos(phi), np.cos(psi)*np.sin(theta)*np.cos(phi) + np.sin(psi)*np.sin(phi)],
            [np.sin(psi)*np.cos(theta), np.sin(psi)*np.sin(theta)*np.sin(phi) + np.cos(psi)*np.cos(phi), np.sin(psi)*np.sin(theta)*np.cos(phi) - np.cos(psi)*np.sin(phi)],
            [-np.sin(theta), np.cos(theta)*np.sin(phi), np.cos(theta)*np.cos(phi)]
        ])
        
        # Fuerza total en el eje Z del mundo (empuje en B rotado a W)
        Acc_z_body = T / self.m
        Acc_W = R_BW @ np.array([0, 0, Acc_z_body])
        
        dvx = Acc_W[0]
        dvy = Acc_W[1]
        dvz = Acc_W[2] - self.g # Aceleración de la gravedad

        # 4. Dinámica Rotacional (Velocidad angular y Actitud)
        
        # EDOs para la velocidad angular (p, q, r)
        dp = (self.Iyy - self.Izz) / self.Ixx * q * r + Tau_phi / self.Ixx
        dq = (self.Izz - self.Ixx) / self.Iyy * p * r + Tau_theta / self.Iyy
        dr = (self.Ixx - self.Iyy) / self.Izz * p * q + Tau_psi / self.Izz

        # EDOs para los ángulos de Euler (phi, theta, psi)
        dphi = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
        dtheta = q * np.cos(phi) - r * np.sin(phi)
        dpsi = q * np.sin(phi) / np.cos(theta) + r * np.cos(phi) / np.cos(theta)

        # 5. Retornar el vector de derivadas
        dXdt = [
            vx, vy, vz,         # d(Posición) = Velocidad lineal
            dvx, dvy, dvz,      # d(Velocidad lineal) = Aceleración
            dphi, dtheta, dpsi, # d(Actitud) = Velocidad angular transformada
            dp, dq, dr          # d(Velocidad angular) = Torque / Inercia
        ]
        return np.array(dXdt)

    def step(self, X0, U, dt):
        """ Integra el sistema un paso de tiempo dt """
        sol = solve_ivp(
            self._state_derivative,
            [0, dt],
            X0,
            args=(U,),
            method='RK45'
        )
        return sol.y[:, -1] # Retorna el estado final