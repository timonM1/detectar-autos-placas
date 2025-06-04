import cv2
import torch
from ultralytics import YOLO


class DetectorPlacas:
    def __init__(self, modelo_placas_path):
        """Detector de placas independiente"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_placas = YOLO(modelo_placas_path).to(self.device)
        self.conf_threshold = 0.35  # Umbral de confianza para evitar objetos que el modelo cree que son placas pero con muy poca seguridad
        
    def detectar_placa(self, crop_auto):
        """Detecta placa en un crop de auto"""
        try:
            # Redimensionar crop para mejor detección
            h, w = crop_auto.shape[:2]
            if h < 100 or w < 100:
                return None
                
            # Detección de placa
            results = self.model_placas(
                crop_auto,
                verbose=False,
                conf=self.conf_threshold,
                iou=0.4,
                max_det=1,  # Maximo 1 detecciones por crop
                half=True if self.device == 'cuda' else False,
                device=self.device
            )[0]
            
            if len(results.boxes) > 0:
                box = results.boxes.xyxy[0].cpu().numpy()
                conf = float(results.boxes.conf[0].cpu().numpy())
                x1, y1, x2, y2 = map(int, box)

                if x2 > x1 and y2 > y1:
                    return {
                        'box': (x1, y1, x2, y2),
                        'conf': conf,
                        'crop': crop_auto[y1:y2, x1:x2]
                    }
                    
        except Exception as e:
            print(f"Error detectando placa: {e}")
            
        return None