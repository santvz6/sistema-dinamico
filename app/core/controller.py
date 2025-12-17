# app/core/controller.py

import numpy as np

class CascadedController:
    """
    Controlador en Cascada: 
    1. Bucle Externo: Posición (genera ángulos deseados)
    2. Bucle Interno: Actitud (genera torques)
    """
    def __init__(self, dynamics):
        self.dynamics = dynamics
        self.mass = dynamics.m
        self.g = dynamics.g
        

        ### Ganancias de Actitud (Bucle Interno: Controla phi, theta, psi)
        # Este bucle controla la orientación del dron (roll, pitch, yaw)
        # Se suele poner más agresivo para que el dron responda rápido a cambios de ángulo
        self.Kp_att = np.array([2.5, 2.5, 2.5]) * self.mass / dynamics.m_ref
        # Kp_att: Ganancia proporcional de actitud
        #  - Controla cuánto responde el dron a un error de ángulo
        #  - Valores altos: respuesta rápida, pero puede generar oscilaciones
        self.Kd_att = np.array([0.5, 0.5, 0.5]) * self.mass / dynamics.m_ref
        # Kd_att: Ganancia derivativa de actitud
        #  - Controla la respuesta frente a la velocidad angular
        #  - Ayuda a amortiguar oscilaciones y suavizar movimientos

        #### Ganancias de Posición (Bucle Externo: Controla x, y, z)
        # Este bucle controla la posición del dron en el espacio (x, y, z)
        # Se suele poner menos agresivo que el bucle interno, porque la posición cambia más lentamente
        self.Kp_pos = np.array([1.0, 1.0, 1.0]) * self.mass / dynamics.m_ref
        # Kp_pos: Ganancia proporcional de posición
        #  - Simple: Determina qué tan agresivo es el dron al acercarse al objetivo
        self.Kd_pos = np.array([1.0, 1.0, 1.0]) * self.mass / dynamics.m_ref
        # Kd_pos: Ganancia derivativa de posición
        #  - Simple: Determina cuánto frena el dron ante la velocidad

    def control_position(self, X, P_des):
        """ 
        Bucle Externo: Calcula los ángulos de actitud deseados (phi_d, theta_d) 
        y el Empuje total (T).
        """
        # Estado actual: [x, y, z, vx, vy, vz]
        P = X[0:3]
        V = X[3:6]
        
        # Errores
        e_pos = P_des - P
        e_vel = -V # Asumimos velocidad deseada V_des = [0, 0, 0]
        
        F_c = (
            self.Kp_pos * e_pos +
            self.Kd_pos * e_vel 
        )
        
        ### Calculo del Empuje Total (T_required) y ángulos deseados
        # Eje Z (Empuje Total)
        # Incluye la gravedad (estabiliza el hover)
        phi = X[6]
        theta = X[7]

        T_required = self.mass * (self.g + F_c[2]) / (np.cos(phi) * np.cos(theta))
        
        # Ejes X e Y (Ángulos deseados)
        # La forma de calcularlos es una aproximación para pequeños ángulos
        phi_des = np.clip(-F_c[1] / (self.mass * self.g), -0.2, 0.2)    # Pitch deseado para moverse en X
        theta_des = np.clip(F_c[0] / (self.mass * self.g), -0.2, 0.2)   # Roll deseado para moverse en Y
        psi_des = 0.0   # Yaw deseado (lo mantenemos fijo dado que no es totalmente necesario)

        return T_required, np.array([phi_des, theta_des, psi_des])

    def control_attitude(self, X, ref_angles):
        """ 
        Bucle Interno: Aplica PD a los errores de actitud para generar torques (Tau).
        """
        # Estado actual: [phi, theta, psi, p, q, r]
        E = X[6:9]
        W = X[9:12]
        
        # Errores
        e_att = ref_angles - E
        e_w = -W # Asumimos velocidad angular deseada W_des = [0, 0, 0]
        
        # Torques de control (Tau)
        Tau = self.Kp_att * e_att + self.Kd_att * e_w

        return Tau