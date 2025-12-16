# visualizer.py
# Visualización de Trayectorias y Errores

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation # <--- NUEVA IMPORTACIÓN
import numpy as np
import os

class Plotter:
    """ Herramientas para graficar la simulación del Quadrotor. """
    
    def __init__(self, plot_dir:str):
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


    def animate_3d_trajectory(self, hist_pos, target_pos, time_step):
        """
        Genera una animación 3D de la trayectoria de posición del dron.
        Necesita el historial de posición y el paso de tiempo (dt).
        """
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Configurar límites y etiquetas (similar a plot_3d_trajectory)
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
        ax.set_title('Quadrotor 3D Animation')

        # Dibujar la posición objetivo
        ax.scatter(target_pos[0], target_pos[1], target_pos[2], color='r', marker='*', s=200, label='Target Position')
        
        # Elementos que se van a animar
        line, = ax.plot([], [], [], 'b', alpha=0.5, label='Trajectory Trace') 
        point, = ax.plot([], [], [], 'go', markersize=10, label='Drone Position') # El dron en sí
        
        # Función de inicialización
        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
            return line, point

        # Función de actualización para cada frame
        def update(frame):
            # Trazar el camino recorrido hasta el frame actual
            x_data = hist_pos[:frame, 0]
            y_data = hist_pos[:frame, 1]
            z_data = hist_pos[:frame, 2]
            
            line.set_data(x_data, y_data)
            line.set_3d_properties(z_data)
            
            # Posición actual del dron
            point.set_data([hist_pos[frame, 0]], [hist_pos[frame, 1]])
            point.set_3d_properties([hist_pos[frame, 2]])
            
            return line, point

        # Crear la animación
        # interval = dt * 1000 ms
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=range(0, len(hist_pos), 4),
            interval=time_step * 1000,
            blit=False
        )

        plt.legend()
        plt.show()