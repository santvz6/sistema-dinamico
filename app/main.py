# app/main.py

import numpy as np

from logger_config import logger
from config import PLOT_DIR
from core import QuadrotorDynamics, CascadedController, Plotter

class Drone:
    def __init__(self, T_end:float, dt:float):
        self.quadrotor= QuadrotorDynamics(L=0.2)
        self.controller = CascadedController(self.quadrotor)
        self.plotter = Plotter(dynamics=self.quadrotor, plot_dir=PLOT_DIR)

        # Simulation Time
        self.T_end = T_end   # Duración total de la simulación (s)
        self.dt = dt         # Paso de integración (s)
        self.time = np.arange(0, self.T_end, self.dt)
        self.steps = len(self.time)

        # Waypoints
        self.waypoints_detection = 1 # metros
        self.waypoints_times = []
        self.waypoints = [
            np.array([0.0, 0.0, 10.0]),
            np.array([5.0, 0.0, 5.0]),
            np.array([0.0, 5.0, 0.0]),
            np.array([0.0, 0.0, 0.0])
        ]
        self.current_wp_idx = 0
        self.TARGET_POS = self.waypoints[self.current_wp_idx]

        # Estado inicial = [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
        self.X_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, -0.05, 0.0, 0.0, 0.0, 0.0])

    def run(self, pos_subsampling):
        history = np.zeros((self.steps, self.X_state.shape[0]))
        U_history = np.zeros((self.steps, 4)) # U = F1 + F2 + F3 + F4
        hist_time = np.zeros(self.steps)
        hist_T = np.zeros(self.steps)

        logger.debug("Iniciando simulación...")
        for i in range(self.steps):
            t = self.time[i]
 
            target_distance = np.linalg.norm(self.TARGET_POS - self.X_state[0:3])
            if target_distance < self.waypoints_detection: 
                if self.current_wp_idx < len(self.waypoints) - 1:
                    
                    if t not in self.waypoints_times: # Evitamos duplicados en el mismo paso
                        self.waypoints_times.append(t)
                    
                    self.current_wp_idx += 1
                    self.TARGET_POS = self.waypoints[self.current_wp_idx]
                    logger.info(f"Punto alcanzado! Rumbo al Waypoint {self.current_wp_idx}")
                else:
                    # Circuito infinito
                    self.current_wp_idx = 0
                    self.TARGET_POS = self.waypoints[self.current_wp_idx]
            
            # Control de Posición
            if i % pos_subsampling == 0:    
                T_total, ref_angles = self.controller.control_position(self.X_state, self.TARGET_POS)
            
            # Control de Actitud (retorna torques Tau)
            Tau_ctrl = self.controller.control_attitude(self.X_state, ref_angles)
            
            # Conversión de Control (T, Tau) a Empujes de Rotor (U)
            # Esta es la matriz de asignación de control inversa (M^-1) para la configuración '+'.
            L = self.quadrotor.L
            km = self.quadrotor.km

            U = np.array([
                T_total/4 + Tau_ctrl[0]/(2*L) + 0 + Tau_ctrl[2]/(4*km),          # F1 (Roll+, Yaw+)
                T_total/4 + 0 + Tau_ctrl[1]/(2*L) - Tau_ctrl[2]/(4*km),          # F2 (Pitch+, Yaw-)
                T_total/4 - Tau_ctrl[0]/(2*L) + 0 + Tau_ctrl[2]/(4*km),          # F3 (Roll-, Yaw+)
                T_total/4 + 0 - Tau_ctrl[1]/(2*L) - Tau_ctrl[2]/(4*km)           # F4 (Pitch-, Yaw-)
            ])
            U = np.clip(U, 0, (self.quadrotor.m * self.quadrotor.g))

            # Simulamos la Dinámica
            self.X_state = self.quadrotor.step(self.X_state, U, self.dt)

            # Almacenamos los Históricos
            history[i, :] = self.X_state
            U_history[i, :] = U
            hist_T[i] = np.sum(U)
            hist_time[i] = t

        logger.debug(
            f"Simulación completada. Posición final: X={self.X_state[0]:.2f} m, "
            f"Y={self.X_state[1]:.2f} m, Z={self.X_state[2]:.2f} m"
        )

        self.visualize(history, hist_time, hist_T)
            
            
    def visualize(self, history, hist_time, hist_T):
        # Desempaquetar datos para visualización
        hist_pos = history[:, 0:3]
        hist_vel = history[:, 3:6]
        hist_att = history[:, 6:9]

        logger.debug("Generando Gráficas de Trayectoria y Errores...")
        
        # Animación de la Trayectoria
        self.plotter.animate_3d_trajectory(
            hist_pos=hist_pos,
            hist_vel=hist_vel,
            hist_att=hist_att,
            hist_T=hist_T,
            waypoints=self.waypoints,
            time_step=self.dt,
            filename="3d_trajectory.gif"
        )

        # Visualización de la Trayectoria
        self.plotter.plot_3d_trajectory(hist_pos, self.waypoints, filename="3d_trajectory.png", show=False)

        # Visualización de la Estabilidad
        self.plotter.plot_2d_errors(hist_time, hist_att, hist_pos, self.waypoints, self.waypoints_times, filename="2d_errors.png", show=True)
        

if __name__ == "__main__":
    dt = 0.01
    drone = Drone(T_end=20, dt=dt)

    # Podemos utilizar el sistema en cascada usando varias Frecuencias
    # Un dron suele utilizar frecuenicas con valores de
    # 400Hz-2000Hz para bucles internos y 50-250Hz para bucles externos
    # Nosotros usamos para bucles internos: f = 1 / dt = 1 / 0.01 = 100Hz
    # Y para bucle externo usamos: f_int / pos_subsampling

    # He optado por no utilizar este sistema pero podemos 
    # obtener buenos resultados con valores cercanos a 50
    drone.run(pos_subsampling=0)