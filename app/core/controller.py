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
        # Típicamente más agresivas
        self.Kp_att = np.array([2.5, 2.5, 2.5]) # Kp Roll, Pitch, Yaw
        self.Kd_att = np.array([0.5, 0.5, 0.5]) # Kd Roll, Pitch, Yaw
        
        #### Ganancias de Posición (Bucle Externo: Controla x, y, z)
        # Típicamente menos agresivas
        self.Kp_pos = np.array([0.5, 0.5, 1.0]) # Kp x, y, z
        self.Kd_pos = np.array([0.3, 0.3, 0.8]) # Kd x, y, z


        self.Ki_pos = np.array([0.05, 0.05, 0.1])
        self.e_pos_int = np.zeros(3)
        
    def control_position(self, X, P_des, dt):
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
        
        # Fuerza de control requerida
        self.e_pos_int += e_pos * dt
        self.e_pos_int = np.clip(self.e_pos_int, -2.0, 2.0)

        F_c = (
            self.Kp_pos * e_pos +
            self.Kd_pos * e_vel +
            self.Ki_pos * self.e_pos_int
        )
        
        # Calculo del Empuje Total (T) y ángulos deseados
        
        # Eje Z (Empuje Total)
        # Incluye la gravedad (estabiliza el hover)
        phi = X[6]
        theta = X[7]

        T_required = self.mass * (self.g + F_c[2]) / (np.cos(phi) * np.cos(theta))
        T = np.clip(T_required, 0, 20)
      
        
        # Ejes X e Y (Ángulos deseados) 
        # La forma de calcularlos es una aproximación para pequeños ángulos
        phi_des = np.clip(-F_c[1] / (self.mass * self.g), -0.2, 0.2)     # Pitch deseado para moverse en X
        theta_des = np.clip(F_c[0] / (self.mass * self.g), -0.2, 0.2)  # Roll deseado para moverse en Y
        psi_des = 0.0   # Yaw deseado (lo mantenemos fijo dado que no es totalmente necesario)

        return T, np.array([phi_des, theta_des, psi_des])

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
        
        # 1. Torques de control (Tau)
        Tau = self.Kp_att * e_att + self.Kd_att * e_w
        
        # 2. Conversión de Torque/Empuje (T, Tau) a comandos de motor (U)
        # T (Empuje Total) se obtiene del control_position
        # U = [F1, F2, F3, F4]
        
        # Esta es la matriz de control inversa (simplificada)
        #T_ctrl, Tau_ctrl_arr = Tau[0], Tau[1], Tau[2] # Tau solo para Roll, Pitch, Yaw
        
        # Nota: La conversión completa se hará en el script principal para simplicidad, 
        # aquí solo devolvemos los torques de control.
        return Tau