# app/core/dynamics.py

import numpy as np
from scipy.integrate import solve_ivp


class QuadrotorDynamics:
    """
    Modelado de la dinámica no lineal de un Quadrotor (12 estados).
    """
    def __init__(self, L=0.5):
        # Refrencias a un dron original
        self.m_ref = 0.5
        self.L_ref = 0.25 
        self.kf_ref = 1.0

        # Parámetros Físicos
        self.L = L   # Distancia del centro al rotor (m)
        self.m = self.m_ref * (self.L / self.L_ref)**2   # Masa (kg)
        self.g = 9.81   # Gravedad (m/s^2)
        self.m_rotor = self.m * 0.25 / 4 # Masa de cada rotor (kg)
        self.Ixx = 2 * self.m_rotor * self.L**2  # Momento de inercia x 
        self.Iyy = 2 * self.m_rotor * self.L**2  # Momento de inercia y
        self.Izz = 4 * self.m_rotor * self.L**2  # Momento de inercia z
        self.I = np.array([self.Ixx, self.Iyy, self.Izz])
        self.kf = self.kf_ref * (self.L / self.L_ref)**2 # Coeficiente de fuerza/empuje
        self.km = 0.05 # Coeficiente de momento/arrastre
        self.k_drag = 0.75  # Resistencia al aire
        
    def _state_derivative(self, t, X, U):
        """
        Calcula el vector de derivadas de estado (dX/dt).
        X = [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r] (12 estados)
        U = [F1, F2, F3, F4] (Fuerzas/Empuje de los 4 rotores)
        """
        
        ### Decodificacion de estados
        x, y, z = X[0], X[1], X[2]          # Posición
        vx, vy, vz = X[3], X[4], X[5]       # Velocidad Lineal        
        phi, theta, psi = X[6], X[7], X[8]  # Ángulos de Euler       
        p, q, r = X[9], X[10], X[11]        # Velocidad Angular (p, q, r)

        ### Variables de Control (Empujes individuales y agregados)
        # Hemos modelado un Dron con Quadrotor en forrma de +
        # F1: derecha, F2: adelante, F3: izquierda, F4: atrás
        F1, F2, F3, F4 = U[0], U[1], U[2], U[3]

        T = (F1 + F2 + F3 + F4)  # Empuje total hacia arriba (en el cuerpo)
        
        # Torques (Tau)
        Tau_phi = self.L * (F1 - F3)            # Roll - Rodar      (alrededor de x)
        Tau_theta = self.L * (F2 - F4)          # Pitch - Cabeceo   (alrededor de y)
        Tau_psi = self.km * (F1 - F2 + F3 - F4) # Yaw - Derrape     (alrededor de z)

        ### Dinámica Traslacional (Velocidad lineal)
        # Matriz de Rotación de Cuerpo (B) a Mundo (W)
        # R_BW = R_Z(psi) · R_Y(theta) · R_X(phi) 
        # Para entender las rotaciones: https://www.youtube.com/watch?v=uLl_egj9F2M
        R_BW = np.array([
            [np.cos(psi)*np.cos(theta), np.cos(psi)*np.sin(theta)*np.sin(phi) - np.sin(psi)*np.cos(phi), np.cos(psi)*np.sin(theta)*np.cos(phi) + np.sin(psi)*np.sin(phi)],
            [np.sin(psi)*np.cos(theta), np.sin(psi)*np.sin(theta)*np.sin(phi) + np.cos(psi)*np.cos(phi), np.sin(psi)*np.sin(theta)*np.cos(phi) - np.cos(psi)*np.sin(phi)],
            [-np.sin(theta), np.cos(theta)*np.sin(phi), np.cos(theta)*np.cos(phi)]
        ])
        
        ### Aceleración deseada
        # Solo calculamos la aceleración en Z porque es el único lugar donde se genera la fuerza controlada
        # F = ma -> a = F/m; Concretamente: a_z = F_z/m
        Acc_z_body = T / self.m  # aceleración total que el Empuje Total (T) causa, antes de la rotación
        
        # No hay F_x de control: El dron no tiene propulsores laterales.
        # No hay F_y de control: El dron no tiene propulsores delanteros/traseros.
        Acc_W = R_BW @ np.array([0, 0, Acc_z_body]) # aceleración del dron descompuesta en los ejes X, Y, Z.
        


        # Para calcular la velocidad de nuestro dron
        # tg(clip(0.3rad)) * 9.8) / k_drag

        dvx = Acc_W[0] - self.k_drag * vx # Incluimos la resistencia al aire
        dvy = Acc_W[1] - self.k_drag * vy
        dvz = Acc_W[2] - self.g - self.k_drag * vz



        ### Dinámica Rotacional (Velocidad angular y Actitud)       
        # LEYES DE EULER: Describen la aceleración angular (d(p, q, r)/dt).
        # Esto es el equivalente rotacional de F = m*a (Tau = I*alfa).
        
        # EDOs para la velocidad angular (p, q, r)
        # Aceleración Angular = Efecto Secundario[Acoplamiento] + Efecto Directo[Control]
        # [Control] siempre:  alfa = Tau / I 
        
        # ------------------------------------------------------------------------------------------
        # Aceleración de Roll (dp/dt)
        # dp = [Término de Acoplamiento] + [Término de Control]
        dp = (self.Iyy - self.Izz) / self.Ixx * q * r + Tau_phi / self.Ixx
        # [Acoplamiento]: La combinación de giros en Pitch (q) y Yaw (r) genera un torque secundario
        #                  en el eje Roll, forzando un cambio en 'p' (precesión giroscópica).
        # [Control]: El torque de control directo (Tau_phi) dividido por la inercia (Ixx).
        
        # ------------------------------------------------------------------------------------------
        # Aceleración de Pitch (dq/dt)
        # dq = [Término de Acoplamiento] + [Término de Control]
        dq = (self.Izz - self.Ixx) / self.Iyy * p * r + Tau_theta / self.Iyy
        # [Acoplamiento]: La combinación de Roll (p) y Yaw (r) genera un torque secundario en Pitch.
        # [Control]: El torque de control directo (Tau_theta) dividido por la inercia (Iyy).

        # ------------------------------------------------------------------------------------------
        # Aceleración de Yaw (dr/dt)
        # dr = [Término de Acoplamiento] + [Término de Control]
        dr = (self.Ixx - self.Iyy) / self.Izz * p * q + Tau_psi / self.Izz
        # [Acoplamiento]: La combinación de Roll (p) y Pitch (q) genera un torque secundario en Yaw.
        # [Control]: El torque de control directo (Tau_psi) dividido por la inercia (Izz).
        # ------------------------------------------------------------------------------------------


        ### CINEMÁTICA ROTACIONAL: Describen la tasa de cambio de los ángulos de Euler (d(phi, theta, psi)/dt).
        # Convierten las velocidades angulares del cuerpo (p, q, r) en las tasas de cambio de los ángulos del Mundo.
        
        # Tasa de cambio de Roll (dphi/dt)
        dphi = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
        # La tasa de Roll no es solo 'p', depende de los ángulos de Pitch (theta) y Roll (phi).
        
        # Tasa de cambio de Pitch (dtheta/dt)
        dtheta = q * np.cos(phi) - r * np.sin(phi)
        
        # Tasa de cambio de Yaw (dpsi/dt)
        dpsi = q * np.sin(phi) / np.cos(theta) + r * np.cos(phi) / np.cos(theta)
        # PELIGRO: Este cálculo introduce una división por cos(theta). Si el ángulo de Pitch (theta)
        # se acerca a 90 grados, cos(theta) tiende a cero, causando una singularidad
        # (Gimbal Lock) donde Yaw y Roll se vuelven indistinguibles. 


        ### Retorno del vector de derivadas (dX/dt)
        # Este vector es la entrada para el integrador (solve_ivp).
        dXdt = [
            vx, vy, vz,         # 1-3. d(Posición) = Velocidad lineal
            dvx, dvy, dvz,      # 4-6. d(Velocidad lineal) = Aceleración
            dphi, dtheta, dpsi, # 7-9. d(Actitud) = Tasa de cambio de ángulos (Cinemática)
            dp, dq, dr          # 10-12. d(Velocidad angular) = Aceleración angular (Leyes de Euler)
        ]
        return np.array(dXdt)

    def step(self, X0, U, dt):
        """ Integra el sistema un paso de tiempo dt
        Dado el estado actual y los empujes de los motores, dónde estará el dron en el siguiente instante de tiempo. """
        sol = solve_ivp(
            self._state_derivative,
            [0, dt],
            X0,
            args=(U,),
            method="RK45"
        )
        return sol.y[:, -1] # Retorna el estado final