# main.py

import numpy as np

from logger_config import logger
from config import PLOT_DIR
from core import QuadrotorDynamics, CascadedController, Plotter

class Drone:

    def __init__(self, T_end:float):
        
        self.quadrotor= QuadrotorDynamics()
        self.controller = CascadedController(self.quadrotor)
        self.plotter = Plotter(plot_dir=PLOT_DIR)

        # Simulation Time
        self.T_end = T_end   # Duración total de la simulación (s)
        self.dt = 0.01       # Paso de integración (s)
        self.time = np.arange(0, self.T_end, self.dt)
        self.steps = len(self.time)

        # Objetivo de la Misión: Hover en [x=0, y=0, z=5]
        self.TARGET_POS = np.array([0.0, 0.0, 5.0]) 

        # Estado inicial del dron: ligeramente inclinado y en el suelo
        # X = [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
        self.X_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, -0.05, 0.0, 0.0, 0.0, 0.0])

    def run(self, visualize=False):
        history = np.zeros((self.steps, self.X_state.shape[0]))
        U_history = np.zeros((self.steps, 4))
        time_hist = np.zeros(self.steps)

        for i in range(self.steps):
            t = self.time[i]
            
            # Control de Posición
            T_total, ref_angles = self.controller.control_position(self.X_state, self.TARGET_POS)
            
            # Control de Actitud (retorna torques Tau)
            Tau_ctrl = self.controller.control_attitude(self.X_state, ref_angles)
            
            # Conversión de Control (T, Tau) a Empujes de Rotor (U)
            # T_total = F1 + F2 + F3 + F4
            # Tau_phi = L*(F1 - F3); Tau_theta = L*(F2 - F4); Tau_psi = km*(F1 - F2 + F3 - F4)
            # Resuelve el sistema lineal 4x4:
            U = np.array([
                T_total/4 + Tau_ctrl[0]/(2*self.quadrotor.L) - Tau_ctrl[2]/(4*self.quadrotor.km), # F1
                T_total/4 - Tau_ctrl[1]/(2*self.quadrotor.L) + Tau_ctrl[2]/(4*self.quadrotor.km), # F2
                T_total/4 - Tau_ctrl[0]/(2*self.quadrotor.L) - Tau_ctrl[2]/(4*self.quadrotor.km), # F3
                T_total/4 + Tau_ctrl[1]/(2*self.quadrotor.L) + Tau_ctrl[2]/(4*self.quadrotor.km)  # F4
            ])
            U = np.clip(U, 0, 15) # Limitar los comandos (realismo)

            # Almacenamos los Históricos
            history[i, :] = self.X_state
            U_history[i, :] = U
            time_hist[i] = t

            # Simulamos la Dinámica
            self.X_state = self.quadrotor.step(self.X_state, U, self.dt)

        if visualize: self.visualize(history, time_hist)
            
            
    def visualize(self, history, time_hist):
        # Desempaquetar datos para visualización
        hist_pos = history[:, 0:3]
        hist_att = history[:, 6:9]

        logger.debug("Generando Gráficas de Trayectoria y Errores...")
        
        # Visualización de la Trayectoria
        self.plotter.plot_3d_trajectory(hist_pos, self.TARGET_POS, filename="3d_trajectory.png")

        # Visualización de la Estabilidad
        self.plotter.plot_2d_erors(time_hist, hist_att, hist_pos, self.TARGET_POS, filename="2d_errors.png")
        
        logger.debug(f"Simulación completada. Estado final Z: {self.X_state[2]:.2f} m")

if __name__ == "__main__":
    drone = Drone(T_end=15)
    drone.run()