## Dinámica de Enrutamiento en Redes (Internet o Tráfico Urbano)

Este sistema simula cómo la información o el tráfico se mueve a través de una red con capacidad limitada, y cómo las decisiones de enrutamiento (routing) generan congestión y afectan la velocidad.

### 1. Sistema Dinámico: Flujo en Redes (Graph Theory)

El sistema se modela como un grafo dinámico, donde el flujo se rige por la capacidad de los enlaces y las estrategias de enrutamiento.

* **Variables de Estado:**
    * **Grafo:** Nodos (ciudades, routers, intersecciones) y Enlaces (carreteras, cables de red).
    * **Carga de Enlace ($\lambda_i$):** El porcentaje de capacidad utilizado en cada enlace $i$ en un momento dado.
    * **Tiempo de Viaje ($T_i$):** El tiempo que tarda el flujo en atravesar el enlace $i$, que **aumenta dinámicamente** con la carga $\lambda_i$.
* **Reglas Dinámicas (Enrutamiento):**
    1.  **Generación de Demanda:** Se generan nuevos paquetes de información o vehículos en los nodos de origen con destinos aleatorios.
    2.  **Decisión de Enrutamiento:** En cada nodo, el paquete/vehículo elige el siguiente enlace basándose en una métrica dinámica (ej. el camino con el *tiempo de viaje estimado más bajo* en ese instante).
    3.  **Bucle de Retroalimentación:** La elección del "mejor" camino aumenta su carga, lo que a su vez aumenta su tiempo de viaje, haciendo que **otros caminos sean más atractivos** en el siguiente instante.

### 2. Simulación y Visualización (Colab)

* **Enfoque Técnico:** Requiere la implementación de un algoritmo de búsqueda de ruta (ej. Dijkstra o A*) que se recalcula dinámicamente, y la implementación de una función de coste de enlace que dependa de la carga actual (función no lineal).
* **Visualización (Alto Interés):**
    * Una **visualización de red (grafo)** en 2D que muestra el mapa de la red.
    * El **color y el grosor de los enlaces** cambian dinámicamente:
        * **Verde/Fino:** Baja carga, rápido.
        * **Rojo/Grueso:** Alta carga, congestionado (tráfico o latencia alta).
    * La animación mostraría las "olas" de congestión viajando por la red, un efecto visual muy claro de la dinámica de enrutamiento.

### 3. Ventajas para el Trabajo Final

* **Aplicación Directa:** Ingeniería de redes de datos (Internet) o planificación de transporte urbano.
* **Sistema Dinámico Clásico y No Trivial:** La dinámica compleja proviene del **bucle de retroalimentación** entre la decisión local y el estado global. Esto es un sistema dinámico sofisticado que no es una EDO ni una PDE.
* **Demostración de Conocimiento:** Demuestra tu capacidad para simular algoritmos, dinámicas de redes y bucles de retroalimentación no lineales.

---
