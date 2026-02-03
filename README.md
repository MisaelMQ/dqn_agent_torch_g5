# DQN Reinforcement Learning (Stage ROS2)

Paquete ROS 2 (`ament_python`) que implementa un agente **Deep Q-Network (DQN)** en **PyTorch** para navegación autónoma en el simulador **Stage**.  
El objetivo es aprender políticas de navegación basadas en **LiDAR + odometría**, con *reward shaping*, *curriculum learning* y manejo explícito de reinicios de odometría.

---

## Requisitos

### Software
- Ubuntu 22.04 (recomendado)
- ROS 2 Jazzy
- Python ≥ 3.10
- PyTorch (CPU o CUDA)
- stage_ros2

### Paquetes ROS necesarios
- rclpy
- geometry_msgs
- nav_msgs
- sensor_msgs
- std_srvs

---

## Entorno de simulación (OBLIGATORIO)

Antes de **entrenar o evaluar**, se debe lanzar el entorno de Stage:

```bash
ros2 launch stage_ros2 demo.launch.py world:=cave use_stamped_velocity:=false
```

Este entorno proporciona:
- `/base_scan` (LaserScan)
- `/odom` o `/ground_truth`
- `/cmd_vel`
- Servicio `/reset_positions`

---

## Estructura del paquete

```text
dqn_stage_nav_torch/
├── dqn_stage_nav_torch/
│   ├── __init__.py
│   ├── torch_dqn_agent.py
│   ├── state_processor.py
│   ├── train_node.py
│   ├── eval_node.py
│   └── odom_reset_wrapper.py
├── launch/
│   └── train.launch.py
├── resource/
│   └── dqn_stage_nav_torch
├── test/
│   ├── test_flake8.py
│   ├── test_pep257.py
│   └── test_copyright.py
├── package.xml
├── setup.py
├── setup.cfg
└── README.md
```

---

## Descripción de archivos

### `torch_dqn_agent.py`
Implementa el agente DQN:
- Arquitectura **CNN/MLP híbrida**
- Dueling DQN
- Double DQN
- Replay Buffer
- Soft update del target network
- Guardado y carga de checkpoints

Clases principales:
- `TorchDQNConfig`
- `QNetwork`
- `ReplayBuffer`
- `TorchDQNAgent`

---

### `state_processor.py`
Encargado de:
- Discretizar el LiDAR en bins
- Normalizar distancias
- Construir el vector de estado final

El estado incluye:
- LiDAR reducido
- Distancia y ángulo al objetivo
- Velocidades previas
- Información de seguridad (distancia mínima a obstáculo)

---

### `odom_reset_wrapper.py`
Nodo **crítico** para entrenamiento y evaluación.

Funciones:
- Escucha `/odom`
- Publica `/odom/sim` con origen reiniciado
- Permite reiniciar la odometría usando un servicio ROS

Esto evita que la red vea posiciones absolutas inconsistentes entre episodios.

---

### `train_node.py`
Nodo principal de entrenamiento.

Responsabilidades:
- Gestión de episodios
- Curriculum learning (easy → medium → hard)
- Reward shaping detallado
- Detección de colisión, estancamiento, timeout
- Logging a CSV
- Guardado de modelos

Rewards considerados:
- Penalización por paso
- Progreso hacia el objetivo
- Orientación al objetivo
- Proximidad a obstáculos
- Penalización por giro excesivo
- Bonus cerca del objetivo
- Penalización por alejarse demasiado

---

### `eval_node.py`
Nodo de evaluación.

- Usa un modelo entrenado (`.pth`)
- No entrena ni modifica pesos
- Ejecuta la política greedy (sin exploración)
- Permite observar el comportamiento aprendido

---

## Entrenamiento

1. Lanzar el entorno Stage (obligatorio):

```bash
ros2 launch stage_ros2 demo.launch.py world:=cave use_stamped_velocity:=false
```

2. Ejecutar el entrenamiento:

```bash
ros2 launch dqn_stage_nav_torch train.launch.py
```

El entrenamiento:
- Guarda checkpoints en `models/`
- Registra métricas en `training_log.csv`
- Puede reanudarse automáticamente desde el último checkpoint

---

## Evaluación

Para **evaluar**, NO se usa launch.

Solo se ejecutan los siguientes nodos:

```bash
ros2 run dqn_stage_nav_torch odom_reset_wrapper.py
ros2 run dqn_stage_nav_torch eval_node.py
```

Orden recomendado:
1. Asegurarse de que Stage esté corriendo
2. Ejecutar `odom_reset_wrapper.py`
3. Ejecutar `eval_node.py`

Durante la evaluación:
- No hay exploración (`epsilon = 0`)
- El robot sigue estrictamente la política aprendida
- No se modifican los pesos

---

## Build del paquete

Desde tu workspace:

```bash
colcon build --packages-select dqn_stage_nav_torch
source install/setup.bash
```

---

## Notas importantes

- El wrapper de odometría es **obligatorio**
- El entorno debe ser siempre el mismo durante train y eval
- Cambiar el mundo (`world:=cave`) invalida modelos entrenados
- Los parámetros de reward están pensados para **Stage + LiDAR 2D**
- El sistema no usa mapa global ni SLAM

---

## Autor

Proyecto académico / experimental  
Uso libre para investigación y aprendizaje
