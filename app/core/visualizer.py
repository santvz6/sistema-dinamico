# visualizer.py
# Visualización de Trayectorias y Errores

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation # <--- NUEVA IMPORTACIÓN
import numpy as np
import os

class Plotter:
    """ Herramientas para graficar la simulación del Quadrotor. """
    
    def __init__(self, dynamics, plot_dir:str, arm_length=0.35):
        self.dynamics = dynamics
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)

        plt.style.use("seaborn-v0_8")


    def plot_3d_trajectory(self, hist_pos, target_pos, filename=None):
        """ Visualiza la trayectoria 3D. (Estática) """
        
        # ... (código existente para la gráfica estática 3D)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Trayectoria simulada
        ax.plot(hist_pos[:, 0], hist_pos[:, 1], hist_pos[:, 2], label='Trayectoria del Dron', color='blue')
        
        # Punto objetivo (final de la misión)
        ax.scatter(target_pos[0], target_pos[1], target_pos[2], c='red', marker='o', s=100, label='Objetivo')
        
        # Configuración de límites (mejorada para que la trayectoria se vea centrada)
        max_range = np.array([hist_pos[:, 0].max()-hist_pos[:, 0].min(), 
                              hist_pos[:, 1].max()-hist_pos[:, 1].min(), 
                              hist_pos[:, 2].max()-hist_pos[:, 2].min()]).max() / 2.0 + 0.5

        mid_x = (hist_pos[:, 0].max() + hist_pos[:, 0].min()) * 0.5
        mid_y = (hist_pos[:, 1].max() + hist_pos[:, 1].min()) * 0.5
        mid_z = (hist_pos[:, 2].max() + hist_pos[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Trayectoria 3D del Quadrotor Controlado')
        ax.legend()
        if filename: plt.savefig(os.path.join(self.plot_dir, filename))
        plt.show()


    def plot_2d_erors(self, hist_time, hist_att, hist_pos, target_pos, filename=None):
        """ Visualiza la evolución de los errores de control. """
        
        # Errores de Posición (X, Y, Z)
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        pos_labels = ['X', 'Y', 'Z']
        
        for i in range(3):
            axs[i].plot(hist_time, hist_pos[:, i] - target_pos[i], label=f'Error {pos_labels[i]}')
            axs[i].axhline(0, color='red', linestyle='--', linewidth=0.8)
            axs[i].set_ylabel(f'Error {pos_labels[i]} (m)')
            axs[i].grid(True)
        axs[0].set_title('Errores de Posición vs. Tiempo')
        axs[-1].set_xlabel('Tiempo (s)')
        
        # Errores de Actitud (Roll, Pitch, Yaw)
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        att_labels = ['Roll ($\phi$)', 'Pitch ($\\theta$)', 'Yaw ($\psi$)']
        
        for i in range(3):
            # Asumimos que la referencia de actitud es 0 (para hover estable)
            # Nota: Los errores de actitud deberían compararse con ref_angles, 
            # pero como en main.py solo se guardan los ángulos absolutos y
            # la referencia es 0 para el hover, usamos hist_att[:, i].
            axs[i].plot(hist_time, hist_att[:, i], label=f'{att_labels[i]}')
            axs[i].axhline(0, color='red', linestyle='--', linewidth=0.8)
            axs[i].set_ylabel(f'Ángulo {att_labels[i]} (rad)')
            axs[i].grid(True)
        axs[0].set_title('Ángulos de Actitud vs. Tiempo')
        axs[-1].set_xlabel('Tiempo (s)')
        
        plt.tight_layout()
        if filename: plt.savefig(os.path.join(self.plot_dir, filename))
        plt.show()

    def animate_3d_trajectory(self, hist_pos, hist_vel, hist_att, hist_T,
                           target_pos, time_step, filename):
        """
        Genera una animación 3D de la trayectoria de posición del dron.
        Necesita el historial de posición y el paso de tiempo (dt).
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
        ax.set_title("Quadrotor 3D Animation")

        # Objetivo
        ax.scatter(*target_pos, color="r", marker="*", s=200, label="Target")

        # ================== TRAYECTORIA ==================
        traj_line, = ax.plot([], [], [], "b", alpha=0.5, label="Trajectory")
        drone_point, = ax.plot([], [], [], "go", markersize=6)

        # ================== DRON ==================
        L = self.dynamics.L

        arms_body = np.array([
            [[-L, 0, 0], [ L, 0, 0]],
            [[0, -L, 0], [0,  L, 0]]
        ])

        arm_lines = [
            ax.plot([], [], [], "k", lw=3)[0],
            ax.plot([], [], [], "k", lw=3)[0]
        ]

        # ================== TEXTO ==================
        info_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

        # ================== INIT ==================
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

        # ================== UPDATE ==================
        def update(frame):
            pos = hist_pos[frame]
            phi, theta, psi = hist_att[frame]

            # Trayectoria
            traj_line.set_data(hist_pos[:frame, 0], hist_pos[:frame, 1])
            traj_line.set_3d_properties(hist_pos[:frame, 2])

            # Centro del dron
            drone_point.set_data([pos[0]], [pos[1]])
            drone_point.set_3d_properties([pos[2]])

            # Rotación
            R = rotation_matrix(phi, theta, psi)

            for i, arm in enumerate(arms_body):
                arm_world = (R @ arm.T).T + pos
                arm_lines[i].set_data(arm_world[:, 0], arm_world[:, 1])
                arm_lines[i].set_3d_properties(arm_world[:, 2])

            # Texto
            vx, vy, vz = hist_vel[frame]
            T = hist_T[frame]

            info_text.set_text(
                f"Vel: [{vx:.2f}, {vy:.2f}, {vz:.2f}]\n"
                f"φ={phi:.2f}, θ={theta:.2f}, ψ={psi:.2f}\n"
                f"T={T:.2f}"
            )

            return traj_line, drone_point, *arm_lines, info_text

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=range(0, len(hist_pos), 4),
            init_func=init,
            interval=time_step * 1000,
            blit=False
        )

        plt.legend()
        ani.save(os.path.join(self.plot_dir, filename), writer="pillow", fps=30)
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