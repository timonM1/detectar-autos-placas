import torch
from ultralytics import YOLO


class DetectorVehiculos:
    def __init__(self, modelo_path, device='cpu'):
        self.device = device
        self.model = YOLO(modelo_path).to(self.device)
        self.conf_threshold = 0.4
        self.min_box_size = 100 
        self.min_auto_size = 100
        self.colores = {
            'car': (0, 255, 0),
            'bus': (255, 0, 255),
            'truck': (0, 165, 255),
        }

    def detectar(self, frame, frame_idx):
        results = self.model(
            frame,
            verbose=False,
            classes=[2, 5, 7],  # car, bus, truck
            conf=self.conf_threshold,
            iou=0.5,
            half=True if self.device == 'cuda' else False,
            device=self.device
        )[0]

        vehiculos = []
        if len(results.boxes) > 0:
            for i, (box, cls_id, conf) in enumerate(zip(results.boxes.xyxy.cpu().numpy(),
                                                       results.boxes.cls.cpu().numpy(),
                                                       results.boxes.conf.cpu().numpy())):
                
                clase = results.names[int(cls_id)]
                if clase not in self.colores:
                    continue

                x1, y1, x2, y2 = map(int, box)
                
                # Filtro de tamaño
                if (x2 - x1) < self.min_box_size or (y2 - y1) < self.min_box_size:
                    continue

                auto_id = f"{frame_idx}_{i}_{clase}"
                
                vehiculo_info = {
                    'box': (x1, y1, x2, y2),
                    'clase': clase,
                    'conf': float(conf),
                    'frame_idx': frame_idx,
                    'auto_id': auto_id
                }
                vehiculos.append(vehiculo_info)

        return vehiculos