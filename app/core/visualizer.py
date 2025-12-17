# visualizer.py
# Visualización de Trayectorias y Errores

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation # <--- NUEVA IMPORTACIÓN
import numpy as np
import os

from helpers import prepare_plot_folder


class Plotter:
    """ Herramientas para graficar la simulación del Quadrotor. """
    
    def __init__(self, dynamics, plot_dir:str, arm_length=0.35):
        self.dynamics = dynamics
        os.makedirs(plot_dir, exist_ok=True)
        self.plot_dir = prepare_plot_folder(plot_dir)

        plt.style.use("seaborn-v0_8")


    def plot_3d_trajectory(self, hist_pos, waypoints, filename=None, show=True):
        """ Visualiza la trayectoria 3D estática con todo el circuito. """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Trayectoria simulada
        ax.plot(hist_pos[:, 0], hist_pos[:, 1], hist_pos[:, 2], label='Trayectoria del Dron', color='blue', alpha=0.7)
        
        # Dibujar todos los Waypoints del circuito
        wps = np.array(waypoints)
        ax.scatter(wps[:, 0], wps[:, 1], wps[:, 2], c='red', marker='*', s=150, label='Waypoints')
        ax.plot(wps[:, 0], wps[:, 1], wps[:, 2], 'r--', alpha=0.4, label='Ruta planificada')
        
        # Ajuste de límites dinámicos
        all_points = np.vstack([hist_pos, wps])
        max_range = np.ptp(all_points, axis=0).max() / 2.0 + 0.5
        mid = all_points.mean(axis=0)

        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Circuito Completo y Trayectoria Realizada')
        ax.legend()
        
        if filename: plt.savefig(os.path.join(self.plot_dir, filename))
        if show: plt.show()
        else: plt.close(fig)

    def plot_2d_errors(self, hist_time, hist_att, hist_pos, waypoints, wp_times, filename=None, show=True):
        """ 
        Muestra la evolución de los ángulos y la posición absoluta. 
        Nota: En circuitos, graficar el 'error' contra el punto final es confuso, 
        por lo que graficamos la posición real vs los niveles de los waypoints.
        """
        fig, axs = plt.subplots(6, 1, figsize=(10, 14), sharex=True)
        wps = np.array(waypoints)
        pos_labels = ["X", "Y", "Z"]
        att_labels = ["Roll ($\\phi$)", "Pitch ($\\theta$)", "Yaw ($\\psi$)"]
        
        # Posiciones reales y marcas de Waypoints
        for i in range(6):
            # Dibujamos las líneas verticales de llegada en todos los subplots
            if wp_times:
                for t_reach in wp_times:
                    axs[i].axvline(x=t_reach, color='green', linestyle='--', alpha=0.6, label="WP Reached" if i==0 and t_reach == wp_times[0] else "")

            if i < 3: # Subplots de Posición
                axs[i].plot(hist_time, hist_pos[:, i], label=f"Real {pos_labels[i]}", color='tab:blue')
                # Líneas horizontales de los niveles de los Waypoints
                for wp_val in np.unique(wps[:, i]):
                    axs[i].axhline(wp_val, color="red", linestyle=":", alpha=0.4)
                axs[i].set_ylabel(f"{pos_labels[i]} (m)")
            else: # Subplots de Actitud
                axs[i].plot(hist_time, hist_att[:, i-3], label=f"{att_labels[i-3]}", color='tab:orange')
                axs[i].axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.2)
                axs[i].set_ylabel(f"{att_labels[i-3]} (rad)")
            
            axs[i].grid(True)
            axs[i].legend(loc='upper right', fontsize='small')

        axs[0].set_title("Telemetría de Vuelo: Líneas Verdes = Waypoint Alcanzado")
        axs[-1].set_xlabel("Tiempo (s)")
        plt.tight_layout()
        
        if filename: plt.savefig(os.path.join(self.plot_dir, filename))
        if show: plt.show()
        else: plt.close(fig)
    
    def animate_3d_trajectory(self, hist_pos, hist_vel, hist_att, hist_T,
                            waypoints, time_step, filename):
        """
        Genera una animación 3D del dron recorriendo el circuito de waypoints.
        """

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        # ================== LÍMITES ==================
        max_range = np.ptp(hist_pos, axis=0).max() / 2 + 0.5
        mid = hist_pos.mean(axis=0)

        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Quadrotor Circuit Animation")

        # ================== DIBUJAR CIRCUITO ==================
        wps = np.array(waypoints)
        # Dibujar todos los puntos del circuito
        ax.scatter(wps[:, 0], wps[:, 1], wps[:, 2], color="r", marker="*", s=200, label="Waypoints")
        # Línea punteada que une los puntos del circuito
        ax.plot(wps[:, 0], wps[:, 1], wps[:, 2], "r--", alpha=0.3)

        # ================== TRAYECTORIA REAL ==================
        traj_line, = ax.plot([], [], [], "b", alpha=0.5, label="Trajectory")
        drone_point, = ax.plot([], [], [], "go", markersize=6)

        # ================== ESTRUCTURA DRON ==================
        L = self.dynamics.L
        arms_body = np.array([
            [[-L, 0, 0], [ L, 0, 0]],
            [[0, -L, 0], [0,  L, 0]]
        ])

        arm_lines = [
            ax.plot([], [], [], "k", lw=3)[0],
            ax.plot([], [], [], "k", lw=3)[0]
        ]

        # ================== INFO TEXTO ==================
        info_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

        def init():
            traj_line.set_data([], [])
            traj_line.set_3d_properties([])
            drone_point.set_data([], [])
            drone_point.set_3d_properties([])
            for arm in arm_lines:
                arm.set_data([], [])
                arm.set_3d_properties([])
            info_text.set_text("")
            return traj_line, drone_point, *arm_lines, info_text

        def update(frame):
            pos = hist_pos[frame]
            phi, theta, psi = hist_att[frame]

            # Actualizar línea de trayectoria
            traj_line.set_data(hist_pos[:frame, 0], hist_pos[:frame, 1])
            traj_line.set_3d_properties(hist_pos[:frame, 2])

            # Actualizar posición dron
            drone_point.set_data([pos[0]], [pos[1]])
            drone_point.set_3d_properties([pos[2]])

            # Calcular rotación del dron
            R = rotation_matrix(phi, theta, psi)
            for i, arm in enumerate(arms_body):
                arm_world = (R @ arm.T).T + pos
                arm_lines[i].set_data(arm_world[:, 0], arm_world[:, 1])
                arm_lines[i].set_3d_properties(arm_world[:, 2])

            # Info en pantalla
            vx, vy, vz = hist_vel[frame]
            info_text.set_text(
                f"X, Y, Z: {pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}\n"
                f"Vel: {np.linalg.norm([vx, vy, vz]):.2f} m/s\n"
                f"Angs: φ={phi:.2f}, θ={theta:.2f}, ψ={psi:.2f}"
            )
            return traj_line, drone_point, *arm_lines, info_text

        writer = animation.PillowWriter(fps=20) 
        save_path = os.path.join(self.plot_dir, filename)

        print(f"Grabando y visualizando simultáneamente en: {filename}")
        
        # En lugar de FuncAnimation, usamos un bucle manual con el writer
        with writer.saving(fig, save_path, dpi=100):
            # Saltamos de 5 en 5 como hacías antes
            for frame in range(0, len(hist_pos), 5):
                # Llamamos a tu función update manual
                update(frame) 
                
                # --- ESTAS LÍNEAS HACEN LA MAGIA ---
                plt.draw()       # Dibuja el frame actual en la ventana
                plt.pause(0.01)  # Pausa breve para que Windows actualice la imagen
                writer.grab_frame() # Guarda el frame actual en el archivo
                # ----------------------------------

        plt.show()



def rotation_matrix(phi, theta, psi):
    """
    Matriz de rotación ZYX (Yaw-Pitch-Roll)
    """
    Rz = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi),  np.cos(psi), 0],
        [0, 0, 1]
    ])

    Ry = np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi),  np.cos(phi)]
    ])

    return Rz @ Ry @ Rx