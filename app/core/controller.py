# app/core/controller.py

import numpy as np

class CascadedController:
    """
    Controlador en Cascada: 
    1. Bucle Externo: Posición (genera los ángulos deseados)
    2. Bucle Interno: Actitud (genera los torques)
    """
    def __init__(self, dynamics):
        self.dynamics = dynamics
        self.mass = dynamics.m
        self.g = dynamics.g
        
        #### Ganancias de Posición (Bucle Externo: Controla x, y, z)
        # Se suele poner menos agresivo que el bucle interno        
        self.Kp_pos = np.array([1.0, 1.0, 1.0]) * self.mass / dynamics.m_ref
        # Kp_pos: Ganancia Proporcional
        # Determina qué tan agresivo es el dron al acercarse al objetivo (acelerador)
        self.Kd_pos = np.array([2.5, 2.5, 2.5]) * self.mass / dynamics.m_ref # antes Z=5.0 pq no tenia k_drag
        # Kd_pos: Ganancia Derivativa
        # Determina cuánto frena el dron respecto la velocidad objetivo (freno)
        #
        # nota: al principio el "acelerador" gana por mucho, pero conforme se acerca, 
        # el "acelerador" pierde fuerza y el "freno" se vuelve protagonista.

        ### Ganancias de Actitud (Bucle Interno: Controla phi, theta, psi)
        # Se suele poner más agresivo para que el dron responda rápido a cambios de ángulo
        self.Kp_att = np.array([5.0, 5.0, 5.0]) * dynamics.I
        # Kp_att: Ganancia proporcional de actitud
        #  - Controla cuánto responde el dron a un error de ángulo
        #  - Valores altos: respuesta rápida, pero puede generar oscilaciones
        self.Kd_att = np.array([2.5, 2.5, 2.5]) * dynamics.I
        # Kd_att: Ganancia derivativa de actitud
        #  - Controla la respuesta frente a la velocidad angular
        #  - Ayuda a amortiguar oscilaciones y suavizar movimientos


    def control_position(self, X, P_des, V_des=np.zeros(3)):
        """ 
        Bucle Externo: Calcula los ángulos de actitud deseados (phi_d, theta_d) 
        y el Empuje total (T).
        """

        # Estado actual: [x, y, z, vx, vy, vz, phi, theta, psi]
        P = X[0:3]
        V = X[3:6]
        phi,theta, psi = X[6], X[7], X[8]

        # Errores
        e_pos = P_des - P
        e_vel = V_des - V
        
        # Aceleración lineal deseada
        F_c = (self.Kp_pos * e_pos + self.Kd_pos * e_vel) # shape = (3,)
        
        # Empuje Total deseado
        T = self.mass * (self.g + F_c[2])   # mantiene el dron volando verticalmente
        T /= (np.cos(phi) * np.cos(theta))  # aumenta la potencia para compensar la inclinación
        
        # Angulos deseados
        limit_angle = 0.3 # radianes
        phi_des = np.clip(-F_c[1] / (self.mass * self.g), -limit_angle, limit_angle)    # X: cuánta fuerza queremos hacer en el eje Y (F_c[1]).
        theta_des = np.clip(F_c[0] / (self.mass * self.g), -limit_angle, limit_angle)   # Y: cuánta fuerza queremos hacer en el eje X (F_c[0]).
        psi_des = 0.0   # Yaw deseado (lo mantenemos fijo dado que no es totalmente necesario y simplifica el problema)

        return T, np.array([phi_des, theta_des, psi_des])

    def control_attitude(self, X, ref_angles, W_des=np.zeros(3)):
        """ 
        Bucle Interno: Aplica PD a los errores de actitud para generar torques (Tau).
        """
        # Estado actual: [phi, theta, psi, p, q, r]
        E = X[6:9]
        W = X[9:12]
        
        # Errores
        e_att = ref_angles - E # ref angles: los angulos que queremos
        e_w = W_des  -W
        
        # Torques de control deseados
        Tau = self.Kp_att * e_att + self.Kd_att * e_w

        return Tau