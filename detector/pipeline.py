import cv2
import numpy as np
import torch
import os
import shutil
import time
import threading
from queue import Queue, Empty
from collections import deque

from .detector_placas import DetectorPlacas
from .detector_vehiculos import DetectorVehiculos


class DetectorAsincrono:
    def __init__(self, modelo_vehiculos_path, modelo_placas_path=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️ Usando dispositivo: {self.device.upper()}")
        
        # Cargar detectores
        self.detector_vehiculos = DetectorVehiculos(modelo_vehiculos_path, self.device)
        self.detector_placas = DetectorPlacas(modelo_placas_path) if modelo_placas_path else None
        
        # Configuración optimizada
        self.min_auto_size = 100  # Tamaño mínimo para intentar detectar placa
        
        # Threading para vehículos
        self.frame_queue = Queue(maxsize=5)
        self.detection_queue = Queue(maxsize=10)
        self.running = True
        
        # Threading para placas
        self.placa_queue = Queue(maxsize=10)  # Cola para crops de autos
        self.placas_detectadas = {}  # auto_id -> placa_info
        self.placas_lock = threading.Lock()
        
        # Detecciones actuales para mostrar
        self.current_detections = []
        self.detection_lock = threading.Lock()
        
        # Colores (usar los del detector de vehículos)
        self.colores = self.detector_vehiculos.colores
        
        # Contadores por tipo de vehículo
        self.contadores_vehiculos = {
            'car': 0, 'bus': 0, 'truck': 0
        }
        
        # Estadísticas
        self.frames_procesados = 0
        self.vehiculos_detectados = 0
        self.placas_encontradas = 0
        self.tiempos_procesamiento = deque(maxlen=10)
        self.tiempos_placas = deque(maxlen=10)
        
        # Crear directorios
        reset_folder('crops')
        reset_folder('placas')
        
        self.frames_lentos = set()
    
    def detectar_placas_thread(self):
        """Hilo separado para detección de placas"""
        if not self.detector_placas:
            return
            
        print("Hilo de detección de placas iniciado...")
        
        while self.running:
            try:
                # Obtener crop de auto de la cola
                auto_data = self.placa_queue.get(timeout=0.1)
                auto_id, crop_auto, vehiculo_info = auto_data
                
                start_time = time.time()
                
                # Detectar placa en el crop
                placa_result = self.detector_placas.detectar_placa(crop_auto)
                
                processing_time = (time.time() - start_time) * 1000
                self.tiempos_placas.append(processing_time)
                
                if placa_result:
                    # Guardar crop de placa
                    if placa_result['crop'] is not None:
                        placa_filename = f"placas/placa_{auto_id}_{int(time.time()*1000)}.jpg"
                        cv2.imwrite(placa_filename, placa_result['crop'])
                    
                    # Actualizar registro de placas
                    with self.placas_lock:
                        self.placas_detectadas[auto_id] = {
                            'placa_info': placa_result,
                            'vehiculo_info': vehiculo_info,
                            'frames_vivos': 0,
                            'filename': placa_filename
                        }

                        self.placas_encontradas += 1
                    
                    avg_time = sum(self.tiempos_placas) / len(self.tiempos_placas)
                    print(f"🅿️  Placa detectada en auto {auto_id}: conf={placa_result['conf']:.2f} ({avg_time:.1f}ms)")
                
                self.placa_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                print(f"Error en detección de placas: {e}")
                continue
    
    def detectar_vehiculos_thread(self):
        """Hilo separado para detección de vehículos"""
        print("Hilo de detección de vehículos iniciado...")
        
        
        while self.running:
            try:
                frame_data = self.frame_queue.get(timeout=0.1)
                frame, frame_idx = frame_data
                
                start_time = time.time()
                
                # Detección de vehículos usando el detector
                vehiculos = self.detector_vehiculos.detectar(frame, frame_idx)
                
                processing_time = (time.time() - start_time) * 1000
                self.tiempos_procesamiento.append(processing_time)
                
                if processing_time >= 200:
                    with self.detection_lock:
                        self.frames_lentos.add(frame_idx)
                    
                # Procesar resultados para detección de placas
                for vehiculo in vehiculos:
                    x1, y1, x2, y2 = vehiculo['box']
                    clase = vehiculo['clase']
                    auto_id = vehiculo['auto_id']
                    
                    # Guardar crop general
                    self.guardar_crop_async(frame, (x1, y1, x2, y2), clase, frame_idx)
                    
                    # Si es un auto suficientemente grande, enviarlo para detección de placa
                    if (clase == 'car' and 
                        (x2-x1) >= self.min_auto_size and 
                        (y2-y1) >= self.min_auto_size and 
                        self.detector_placas and 
                        not self.placa_queue.full()):
                        
                        # Expandir crop para mejor detección de placa
                        margin = 10
                        h, w = frame.shape[:2]
                        x1_exp = max(0, x1 - margin)
                        y1_exp = max(0, y1 - margin)
                        x2_exp = min(w, x2 + margin)
                        y2_exp = min(h, y2 + margin)
                        
                        crop_auto = frame[y1_exp:y2_exp, x1_exp:x2_exp]
                        
                        if crop_auto.size > 0:
                            try:
                                self.placa_queue.put_nowait((auto_id, crop_auto, vehiculo))
                            except:
                                pass  # Cola llena, skip
                
                # Actualizar detecciones actuales
                with self.detection_lock:
                    self.current_detections = vehiculos
                    self.frames_procesados += 1
                    self.vehiculos_detectados += len(vehiculos)
                    
                    # Actualizar contadores por tipo
                    for vehiculo in vehiculos:
                        if vehiculo['clase'] in self.contadores_vehiculos:
                            self.contadores_vehiculos[vehiculo['clase']] += 1
                
                # Debug con tipos detectados
                if vehiculos:
                    avg_time = sum(self.tiempos_procesamiento) / len(self.tiempos_procesamiento)
                    tipos_detectados = {}
                    autos_para_placas = 0
                    
                    for v in vehiculos:
                        tipos_detectados[v['clase']] = tipos_detectados.get(v['clase'], 0) + 1
                        if (v['clase'] == 'car' and 
                            (v['box'][2]-v['box'][0]) >= self.min_auto_size and 
                            (v['box'][3]-v['box'][1]) >= self.min_auto_size):
                            autos_para_placas += 1
                    
                    tipos_str = " | ".join([f"{tipo}: {cant}" for tipo, cant in tipos_detectados.items()])
                    placa_str = f" | Autos→Placas: {autos_para_placas}" if autos_para_placas > 0 else ""
                    print(f"📍 Frame {frame_idx}: {tipos_str}{placa_str} ({avg_time:.1f}ms)")
                
                self.frame_queue.task_done()

            except Empty:
                continue
            except Exception as e:
                print(f"Error en detección de vehículos: {e}")
                continue
    
    def guardar_crop_async(self, frame, box, clase, frame_idx):
        """Guardar crop de forma asíncrona"""
        try:
            x1, y1, x2, y2 = box
            margin = 5
            h, w = frame.shape[:2]
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)
            
            crop = frame[y1:y2, x1:x2]
            
            if crop.size > 0 and crop.shape[0] > 30 and crop.shape[1] > 30:
                filename = f"crops/{clase}_{frame_idx}_{int(time.time()*1000)}.jpg"
                cv2.imwrite(filename, crop)
        except Exception as e:
            print(f"Error guardando crop: {e}")
    
    def procesar_video(self, video_path):
        """Procesamiento principal con detección de vehículos y placas"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error abriendo video")
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {total_frames} frames {fps:.1f} FPS")
        if self.detector_placas:
            print("Detección de placas: ACTIVADA")
        
        # Iniciar hilos
        detection_thread = threading.Thread(target=self.detectar_vehiculos_thread)
        detection_thread.daemon = True
        detection_thread.start()
        
        placa_thread = None
        if self.detector_placas:
            placa_thread = threading.Thread(target=self.detectar_placas_thread)
            placa_thread.daemon = True
            placa_thread.start()
        
        frame_idx = 0
        detection_frame_counter = 0
        
        # Control de FPS
        target_fps = min(fps, 30)
        frame_time = 1.0 / target_fps
        last_frame_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC) #Se puede aumentar el tamaño mejorando la visibilidad de las placas, pero se demora mucho mas en procesar cada frame

            # Control de FPS
            current_time = time.time()
            elapsed = current_time - last_frame_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            last_frame_time = time.time()
            
            frame_display = frame.copy()
            
            # Procesar cada x frames
            detection_frame_counter += 1
            if detection_frame_counter >= 5:
                detection_frame_counter = 0
                
                if not self.frame_queue.full():
                    try:
                        self.frame_queue.put_nowait((frame.copy(), frame_idx))
                    except:
                        pass
            
            # Obtener detecciones y placas actuales
            with self.detection_lock:
                detections_to_draw = self.current_detections.copy()
            
            with self.placas_lock:
                placas_a_borrar = []
                placas_actuales = {}

                for auto_id, placa_data in self.placas_detectadas.items():
                    placa_data['frames_vivos'] += 1
                    if placa_data['frames_vivos'] >= 4:
                        placas_a_borrar.append(auto_id)
                    else:
                        placas_actuales[auto_id] = placa_data  # Solo mantener los válidos

                for auto_id in placas_a_borrar:
                    del self.placas_detectadas[auto_id]

                
            # Dibujar detecciones
            for vehiculo in detections_to_draw:
                x1, y1, x2, y2 = vehiculo['box']
                color = self.colores[vehiculo['clase']]
                auto_id = vehiculo.get('auto_id', '')
                
                # Verificar si tiene placa detectada
                tiene_placa = auto_id in placas_actuales
                
                # Dibujar rectángulo (más grueso si tiene placa)
                thickness = 4 if tiene_placa else 3
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, thickness)
                
                # Etiqueta
                conf_text = f"{vehiculo['clase']} {int(vehiculo['conf']*100)}%"
                if tiene_placa:
                    placa_conf = placas_actuales[auto_id]['placa_info']['conf']
                    conf_text += f" | Placa: {int(placa_conf*100)}%"
                
                label_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                
                # Fondo para el texto
                label_color = (0, 255, 255) if tiene_placa else color  # Amarillo si tiene placa
                cv2.rectangle(frame_display, 
                            (x1, y1-30), 
                            (x1 + label_size[0] + 10, y1), 
                            label_color, -1)
                
                cv2.putText(frame_display, conf_text, (x1+5, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            with self.detection_lock:
                es_lento = frame_idx in self.frames_lentos

            if not es_lento:
                cv2.imshow("Deteccion de Vehiculos y Placas", frame_display)
            else:
                print(f"⚠️ Frame {frame_idx} omitido por detección lenta (>200ms)")
            
            # Control de teclado
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            frame_idx += 1
        
        # Cleanup
        self.running = False
        cap.release()
        cv2.destroyAllWindows()
        
        detection_thread.join(timeout=3)
        if placa_thread:
            placa_thread.join(timeout=3)
        
        print("\nProcesamiento completado")
        self.mostrar_estadisticas()
        
        return True
    
    def mostrar_estadisticas(self):
        """Mostrar estadísticas detalladas incluyendo placas"""
        
        print(f"\n📊 ESTADÍSTICAS DETALLADAS:")
        print(f"Frames procesados: {self.frames_procesados}")
        print(f" Total vehículos: {self.vehiculos_detectados}")
        
        if self.detector_placas:
            print(f" Placas detectadas: {self.placas_encontradas}")
        
        # Estadísticas por tipo
        print(f"\n📋 DETECCIONES POR TIPO:")
        for tipo, cantidad in self.contadores_vehiculos.items():
            if cantidad > 0:
                porcentaje = (cantidad / self.vehiculos_detectados * 100) if self.vehiculos_detectados > 0 else 0
                print(f"   {tipo.capitalize():>12}: {cantidad:>4} ({porcentaje:.1f}%)")
        
        print(f"Crops guardados en: crops/")
        if self.detector_placas:
            print(f"Placas guardadas en: placas/")


def reset_folder(folder_path):
    """Resetea una carpeta eliminándola y recreándola"""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  # Elimina toda la carpeta con su contenido
    os.makedirs(folder_path) 
    

def procesar_video(video_path, modelo_vehiculos_path, modelo_placas_path=None):
    """Función principal con detección de vehículos y placas"""
    print("Iniciando detección")
    
    detector = DetectorAsincrono(modelo_vehiculos_path, modelo_placas_path)
    return detector.procesar_video(video_path)