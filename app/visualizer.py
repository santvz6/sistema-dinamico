# visualizer.py
# Visualización de Trayectorias y Errores

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class Plotter:
    """ Herramientas para graficar la simulación del Quadrotor. """
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')

    def plot_3d_trajectory(self, hist_pos, target_pos):
        """ Visualiza la trayectoria 3D. """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Trayectoria simulada
        ax.plot(hist_pos[:, 0], hist_pos[:, 1], hist_pos[:, 2], label='Trayectoria del Dron', color='blue')
        
        # Punto objetivo (final de la misión)
        ax.scatter(target_pos[0], target_pos[1], target_pos[2], c='red', marker='o', s=100, label='Objetivo')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Trayectoria 3D del Quadrotor Controlado')
        ax.legend()
        plt.show()

    def plot_errors_2d(self, hist_time, hist_att, hist_pos, target_pos):
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
            axs[i].plot(hist_time, hist_att[:, i], label=f'{att_labels[i]}')
            axs[i].axhline(0, color='red', linestyle='--', linewidth=0.8)
            axs[i].set_ylabel(f'Ángulo {att_labels[i]} (rad)')
            axs[i].grid(True)
        axs[0].set_title('Ángulos de Actitud vs. Tiempo')
        axs[-1].set_xlabel('Tiempo (s)')
        
        plt.tight_layout()
        plt.show()