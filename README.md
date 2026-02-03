# DQN Reinforcement Learning (Stage ROS2)

Paquete ROS 2 (`ament_python`) que implementa un agente de Deep Reinforcement Learning (DQN con PyTorch) para navegación móvil en el simulador **Stage ROS2**.  
El agente aprende a navegar hacia objetivos evitando obstáculos utilizando información de LIDAR y odometría.

---

## 1. Estructura del paquete

La estructura del paquete sigue las convenciones estándar de ROS 2 para paquetes Python:

dqn_stage_nav_torch/
├── dqn_stage_nav_torch/
│   ├── __init__.py
│   ├── state_processor.py
│   ├── torch_dqn_agent.py
│   ├── train_node.py
│   ├── eval_node.py
│   └── odom_reset_wrapper.py
├── launch/
│   └── train.launch.py
├── models/
│   └── weights...
├── test/
│   ├── test_flake8.py
│   ├── test_pep257.py
│   └── test_copyright.py
├── package.xml
├── setup.py
├── setup.cfg
└── README.md

---

## 2. Descripción de archivos principales

### state_processor.py
Encargado de transformar los datos crudos del LIDAR, la pose del robot y el objetivo en un **vector de estado normalizado** para el agente DQN.

Funciones principales:
- Binning del LIDAR
- Normalización de distancias
- Cálculo de información auxiliar (distancia al objetivo, error angular, velocidades previas)

---

### torch_dqn_agent.py
Implementa el núcleo del algoritmo de aprendizaje por refuerzo.

Incluye:
- Definición de la red Q (MLP o CNN 1D según número de bins)
- Replay Buffer
- Double DQN
- Dueling DQN
- Soft update del target network
- Manejo de epsilon-greedy

Clases principales:
- TorchDQNConfig
- QNetwork
- ReplayBuffer
- TorchDQNAgent

---

### train_node.py
Nodo ROS 2 que controla el **entrenamiento** del agente.

Responsabilidades:
- Recepción de LIDAR y odometría
- Cálculo del estado
- Selección y ejecución de acciones
- Cálculo de recompensas
- Detección de colisiones, éxito, timeout y estados de “stuck”
- Manejo del currículum de objetivos
- Guardado de modelos y métricas en CSV

Este nodo es el corazón del sistema de entrenamiento.

---

### eval_node.py
Nodo ROS 2 utilizado para **evaluar** un modelo previamente entrenado.

Características:
- No entrena la red
- No actualiza el replay buffer
- Ejecuta la política aprendida
- Permite observar el comportamiento del robot en el entorno

---

### odom_reset_wrapper.py
Nodo auxiliar que envuelve la odometría original de Stage para:

- Re-centrar la posición (0,0,0) tras cada reset
- Corregir offsets acumulados
- Publicar una odometría consistente para el agente

Es obligatorio tanto para entrenamiento como para evaluación.

---

## 3. Dependencias principales

- ROS 2 (Jazzy o compatible)
- stage_ros2
- PyTorch
- NumPy
- rclpy
- geometry_msgs
- nav_msgs
- sensor_msgs
- std_srvs

---

## 4. Compilación del paquete

Desde el workspace de ROS 2:

cd ~/ros2_ws
colcon build --packages-select dqn_stage_nav_torch
source install/setup.bash

---

## 5. Entorno de simulación (REQUERIDO)

Antes de **entrenar o evaluar**, es obligatorio lanzar el entorno de simulación Stage:

ros2 launch stage_ros2 demo.launch.py world:=cave use_stamped_velocity:=false

Este entorno proporciona:
- Mundo tipo cueva
- LIDAR
- Odometría
- Servicios de reset

---

## 6. Entrenamiento

Una vez lanzado el entorno Stage, el entrenamiento se ejecuta mediante el launch del paquete:

ros2 launch dqn_stage_nav_torch train.launch.py

Este launch inicia:
- odom_reset_wrapper
- train_node

Durante el entrenamiento:
- Se generan checkpoints en la carpeta `models/`
- Se guarda un archivo CSV con métricas por episodio
- Se conserva el mejor modelo y el último modelo

---

## 7. Evaluación

La evaluación **NO usa launch**.  
Con el entorno Stage ya en ejecución, se lanzan manualmente los nodos:

1) Ejecutar el wrapper de odometría:

ros2 run dqn_stage_nav_torch odom_reset_wrapper.py

2) Ejecutar el nodo de evaluación:

ros2 run dqn_stage_nav_torch eval_node.py

El nodo de evaluación cargará el modelo entrenado y ejecutará la política aprendida sin modificar los pesos.

---

## 8. Flujo completo recomendado

1) Lanzar Stage:
ros2 launch stage_ros2 demo.launch.py world:=cave use_stamped_velocity:=false

2) Entrenar:
ros2 launch dqn_stage_nav_torch train.launch.py

3) Evaluar (en una nueva sesión o tras el entrenamiento):
ros2 run dqn_stage_nav_torch odom_reset_wrapper.py
ros2 run dqn_stage_nav_torch eval_node.py

---

## 9. Notas importantes

- El wrapper de odometría es obligatorio
- El entorno no utiliza mapa global
- La navegación se basa únicamente en LIDAR + odometría
- El sistema está diseñado para experimentación académica y extensión futura

---

## 10. Autor

Proyecto desarrollado con fines académicos para experimentación en aprendizaje por refuerzo aplicado a navegación robótica en ROS 2.
