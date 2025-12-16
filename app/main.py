# main.py
# Ejecución de la Simulación

import numpy as np
from dynamics import QuadrotorDynamics
from controller import CascadedController
from visualizer import Plotter

def main():

    # --- 1. Inicialización ---
    drone = QuadrotorDynamics()
    controller = CascadedController(drone)
    plotter = Plotter()

    # --- 2. Definición de la Misión y Parámetros de Simulación ---
    T_end = 15.0  # Duración total de la simulación (s)
    dt = 0.01     # Paso de integración (s)
    time = np.arange(0, T_end, dt)
    steps = len(time)

    # Objetivo de la Misión: Hover en [x=0, y=0, z=5]
    TARGET_POS = np.array([0.0, 0.0, 5.0]) 

    # Estado inicial del dron: ligeramente inclinado y en el suelo
    # X = [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
    X_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, -0.05, 0.0, 0.0, 0.0, 0.0])

    # Inicialización de Históricos
    history = np.zeros((steps, 12))
    U_history = np.zeros((steps, 4))
    time_hist = np.zeros(steps)


    # --- 3. Bucle de Simulación ---
    for i in range(steps):
        t = time[i]
        
        # 3.1. Bucle Externo: Control de Posición
        T_total, ref_angles = controller.control_position(X_state, TARGET_POS)
        
        # 3.2. Bucle Interno: Control de Actitud (retorna torques Tau)
        Tau_ctrl = controller.control_attitude(X_state, ref_angles)
        
        # 3.3. Conversión de Control (T, Tau) a Empujes de Rotor (U)
        # T_total = F1 + F2 + F3 + F4
        # Tau_phi = L*(F1 - F3); Tau_theta = L*(F2 - F4); Tau_psi = km*(F1 - F2 + F3 - F4)
        # Resuelve el sistema lineal 4x4:
        U = np.array([
            T_total/4 + Tau_ctrl[0]/(2*drone.L) - Tau_ctrl[2]/(4*drone.km), # F1
            T_total/4 - Tau_ctrl[1]/(2*drone.L) + Tau_ctrl[2]/(4*drone.km), # F2
            T_total/4 - Tau_ctrl[0]/(2*drone.L) - Tau_ctrl[2]/(4*drone.km), # F3
            T_total/4 + Tau_ctrl[1]/(2*drone.L) + Tau_ctrl[2]/(4*drone.km)  # F4
        ])
        U = np.clip(U, 0, 15) # Limitar los comandos (realismo)

        # 3.4. Simular la Dinámica
        X_state = drone.step(X_state, U, dt)

        # 3.5. Almacenar Históricos
        history[i, :] = X_state
        U_history[i, :] = U
        time_hist[i] = t
        
        
    # --- 4. Análisis y Visualización ---

    # Desempaquetar datos para visualización
    hist_pos = history[:, 0:3]
    hist_att = history[:, 6:9]

    # 4.1. Visualización de la Trayectoria
    print("Generando Gráficas de Trayectoria y Errores...")
    plotter.plot_3d_trajectory(hist_pos, TARGET_POS)
    # 

    # 4.2. Visualización de la Estabilidad
    plotter.plot_errors_2d(time_hist, hist_att, hist_pos, TARGET_POS)
    # 

    print(f"Simulación completada. Estado final Z: {X_state[2]:.2f}m")

if __name__ == "__main__":
    main()