# Proyecto: Detección de Vehículos y Reconocimiento de Placas

Este proyecto en Python permite detectar vehículos (autos, buses y camiones) en un video, localizar sus placas y luego leerlas utilizando OCR (PaddleOCR). Usa modelos YOLO para detección y OCR para el reconocimiento de texto en las placas.

---

## 📁 Estructura del Proyecto

```
.
├── crops/                  # Recortes de imágenes de vehículos
├── data/                   # Contiene los videos de entrada
│   └── video.mp4
├── detector/               # Módulo de detección (vehículos y placas)
│   ├── lector_placas.py    # Lector OCR de placas
│   ├── detector_placas.py  # Detector de placas con YOLO
│   ├── detector_autos.py   # Detector de vehículos
│   └── pipeline.py         # Funcionamiento principal
├── models/                 # Modelos YOLO entrenados
│   ├── best.pt             # Modelo YOLO para detección de placas
│   └── yolo11n.pt          # Modelo YOLO para detección de vehículos
├── placas/                 # Recortes de placas detectadas (para OCR)
├── main.py                 # Script principal de ejecución
├── requirements.txt        # Dependencias del proyecto
└── README.md               # Documentación del proyecto
```

### 🛠️ Dependencias Internas

```python
import threading
import time
import torch
import cv2
import numpy as np
from queue import Queue, Empty
from collections import deque
```

---

## ▶️ Cómo Usar

1. **Clona este repositorio**
2. Coloca tu video de entrada en:  
   `data/video.mp4`
3. Crea y activa tu entorno virtual (opcional pero recomendado)
4. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
5. Ejecuta el programa:
   ```bash
   python main.py
   ```

---

## 🧠 Estructura del Código

### `main.py`

Coordina todo el flujo del proyecto:

- Detecta vehículos en el video.
- Recorta y guarda las placas.
- Llama a OCR para intentar leerlas.

---

### 📦 Módulo `detector/`

#### `detector_placas.py`

- **Clase:** `DetectorPlacas`
- **Función:** Detecta placas dentro de los recortes de autos, buses y camiones usando un modelo YOLO.

**Método principal:**

```python
detectar_placa(crop_auto)
```

Devuelve:

- Coordenadas de la placa detectada.
- Recorte (`crop`) de la imagen con la placa.
- Nivel de confianza.

---

#### `detector_autos.py`

- **Clase:** `DetectorVehiculos`
- **Función:** Detecta vehículos (carros, buses y camiones) en cada frame del video.

**Método principal:**

```python
detectar(frame, frame_idx)
```

Devuelve una lista con:

- Bounding boxes (`box`)
- Clase (`car`, `bus`, `truck`)
- Confianza de detección
- ID único del vehículo

---

#### `lector_placas.py`

- **Función principal:** `leer_placas_paddle`

Utiliza PaddleOCR para intentar leer el texto en las imágenes de las placas detectadas.

**Parámetros:**

```python
leer_placas_paddle(carpeta_placas: str, debug: bool = False)
```

- `carpeta_placas`: Carpeta donde están los recortes de placas.
- `debug`: Si es `True`, imprime resultados detallados por consola.

---

## Módulo `pipeline.py` – Detección Asíncrona

Este módulo es el **corazón del sistema**: administra la detección de vehículos y placas de forma **asíncrona**, usando **hilos (threads)** y colas para mantener el procesamiento eficiente y no bloquear el flujo de video.

---

### 🚀 Clase `DetectorAsincrono`

Inicializa los modelos y estructuras de control:

```python
detector = DetectorAsincrono(modelo_vehiculos_path, modelo_placas_path)
```

**Componentes principales:**

- Hilos separados para detección de vehículos y placas.
- Colas:
  - `frame_queue`: Frames pendientes de procesar.
  - `placa_queue`: Recortes de autos para enviar al detector de placas.
- Estadísticas y contadores por tipo de vehículo.
- Uso de `deque` para promediar tiempos recientes de detección.

---

### 🔍 Funciones Internas

#### `detectar_vehiculos_thread(self)`

Ejecuta detección de vehículos en frames que llegan por `frame_queue`.  
Guarda recortes de autos y, si son suficientemente grandes, los pasa a `placa_queue`.

#### `detectar_placas_thread(self)`

Lee crops de autos y detecta placas si se cargó un modelo de placas.  
Guarda las placas en disco (`placas/`) y las asocia al `auto_id`.

#### `guardar_crop_async(self, frame, box, clase, frame_idx)`

Guarda en disco un crop del vehículo detectado. Carpeta: `crops/`.

#### `procesar_video(self, video_path)`

Función principal del pipeline:

- Carga el video.
- Inicia los hilos.
- Procesa los frames, los muestra y actualiza detecciones.
- Si un vehículo tiene placa detectada, la etiqueta cambia (color amarillo y confianza).
- Controla FPS para mantener rendimiento estable.
- Finaliza mostrando estadísticas completas.

#### `mostrar_estadisticas(self)`

Imprime en consola:

- Frames procesados
- Vehículos detectados por tipo
- Placas detectadas
- Tiempo promedio de procesamiento

#### `reset_folder(folder_path)`

Limpia una carpeta recreándola desde cero (ej. `crops/`, `placas/`).

---
