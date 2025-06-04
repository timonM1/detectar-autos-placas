import glob
import os
from pathlib import Path
import cv2
import numpy as np
import re
from datetime import datetime

class LectorPlacasPaddle:
    ocr_global = None  # OCR compartido entre instancias

    def __init__(self):
        if LectorPlacasPaddle.ocr_global is None:
            print("🚀 Inicializando PaddleOCR por primera vez...")
            try:
                from paddleocr import PaddleOCR
                print("  Inicializando con configuración mínima...")
                LectorPlacasPaddle.ocr_global = PaddleOCR()
                print("✅ PaddleOCR listo")
            except ImportError:
                print("❌ Error: PaddleOCR no instalado")
                print("Instala con: pip install paddlepaddle paddleocr")
                raise
            except Exception as e:
                print(f"❌ Error crítico inicializando PaddleOCR: {e}")
                raise
        else:
            print("⚡ Reutilizando instancia existente de PaddleOCR")

        self.ocr = LectorPlacasPaddle.ocr_global

    def preprocesar_placa_simple(self, imagen_path):
        """Preprocesamiento simple y efectivo para placas"""
        img = cv2.imread(imagen_path)
        if img is None:
            raise ValueError(f"No se pudo cargar: {imagen_path}")
        
        # 1. Redimensionar para mejor OCR
        h, w = img.shape[:2]
        if w < 400 or h < 100:  # Si es muy pequeña
            scale = max(400/w, 100/h, 3.0)  # Escalar más generosamente
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # 2. Mejorar contraste
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 3. Convertir de vuelta a BGR para PaddleOCR
        img_final = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return img_final

    def leer_placa_con_paddle(self, imagen):
        """Leer placa usando PaddleOCR - versión simple"""
        try:
            # OCR básico
            resultado = self.ocr.ocr(imagen)
            
            if not resultado or not resultado[0]:
                return ""
            
            # Extraer textos
            textos_encontrados = []
            
            for linea in resultado:
                if linea:
                    for deteccion in linea:
                        try:
                            # Formato: [bbox, (texto, confianza)]
                            if len(deteccion) >= 2:
                                texto_info = deteccion[1]
                                if isinstance(texto_info, (tuple, list)) and len(texto_info) >= 2:
                                    texto, confianza = texto_info[0], texto_info[1]
                                else:
                                    texto, confianza = str(texto_info), 0.5
                                
                                # Limpiar texto
                                texto_limpio = re.sub(r'[^A-Z0-9]', '', str(texto).upper())
                                
                                if len(texto_limpio) >= 2 and confianza > 0.2:
                                    textos_encontrados.append({
                                        'texto': texto_limpio,
                                        'confianza': float(confianza)
                                    })
                        except Exception as e:
                            print(f"    Error procesando detección: {e}")
                            continue
            
            if not textos_encontrados:
                return ""
            
            # Tomar el de mayor confianza o combinar si son cortos
            if len(textos_encontrados) == 1:
                return textos_encontrados[0]['texto']
            else:
                # Ordenar por confianza
                textos_encontrados.sort(key=lambda x: x['confianza'], reverse=True)
                
                # Si el mejor es largo, usarlo
                mejor = textos_encontrados[0]['texto']
                if len(mejor) >= 5:
                    return mejor
                
                # Si todos son cortos, intentar combinar los mejores
                texto_combinado = ''.join([t['texto'] for t in textos_encontrados[:3]])
                if 4 <= len(texto_combinado) <= 12:
                    return texto_combinado
                else:
                    return mejor
        
        except Exception as e:
            print(f"    Error en OCR: {e}")
            return ""

    def procesar_imagen_individual(self, path_imagen, carpeta_resultados):
        """Procesar una sola imagen"""
        try:
            nombre_archivo = Path(path_imagen).name
            print(f"\n🔍 Procesando: {nombre_archivo}")
            
            # Solo un tipo de procesamiento
            imagen_procesada = self.preprocesar_placa_simple(path_imagen)
            
            # Guardar imagen procesada en carpeta de resultados
            img_procesada_path = os.path.join(carpeta_resultados, f"procesada_{nombre_archivo}")
            cv2.imwrite(img_procesada_path, imagen_procesada)
            
            # Leer texto
            texto = self.leer_placa_con_paddle(imagen_procesada)
            
            if texto:
                print(f"    ✅ Detectado: '{texto}'")
            else:
                print(f"    ❌ No se detectó texto")
            
            return texto
                
        except Exception as e:
            print(f"❌ Error procesando {path_imagen}: {e}")
            return ""

    def procesar_carpeta_placas(self, carpeta_placas, mostrar_debug=False):
        """Procesar toda la carpeta"""
        # Crear carpeta de resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        carpeta_resultados = os.path.join(carpeta_placas, f"resultados_ocr_{timestamp}")
        os.makedirs(carpeta_resultados, exist_ok=True)
        
        print(f"📁 Resultados se guardarán en: {carpeta_resultados}")
        
        # Buscar imágenes
        extensiones = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
        rutas_imagenes = []
        
        for ext in extensiones:
            rutas_imagenes.extend(glob.glob(os.path.join(carpeta_placas, ext)))
            rutas_imagenes.extend(glob.glob(os.path.join(carpeta_placas, ext.upper())))
        
        if not rutas_imagenes:
            print("❌ No se encontraron imágenes en la carpeta")
            return []
        
        print(f"📂 Procesando {len(rutas_imagenes)} placas...")
        
        resultados = []
        exitosas = 0
        
        for i, ruta in enumerate(rutas_imagenes, 1):
            print(f"\n[{i}/{len(rutas_imagenes)}]", end=" ")
            texto = self.procesar_imagen_individual(ruta, carpeta_resultados)
            
            nombre_archivo = Path(ruta).name
            resultado = {
                'archivo': nombre_archivo,
                'ruta': ruta,
                'texto': texto,
                'exitosa': bool(texto)
            }
            resultados.append(resultado)
            
            if texto:
                exitosas += 1
        
        # Guardar resultados
        self.guardar_resultados(carpeta_resultados, resultados)
        
        # Mostrar resumen
        print(f"\n📊 RESUMEN:")
        print(f"Total procesadas: {len(resultados)}")
        print(f"Exitosas: {exitosas} ({exitosas/len(resultados)*100:.1f}%)")
        print(f"Fallidas: {len(resultados) - exitosas}")
        print(f"📁 Todos los archivos guardados en: {carpeta_resultados}")
        
        return resultados

    def guardar_resultados(self, carpeta_resultados, resultados):
        """Guardar resultados en archivos organizados"""
        
        # 1. Archivo de texto simple
        archivo_txt = os.path.join(carpeta_resultados, "resultados.txt")
        with open(archivo_txt, "w", encoding="utf-8") as f:
            f.write("RESULTADOS DE LECTURA DE PLACAS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total procesadas: {len(resultados)}\n")
            f.write(f"Exitosas: {sum(1 for r in resultados if r['exitosa'])}\n\n")
            
            for resultado in resultados:
                estado = "✅ OK" if resultado['exitosa'] else "❌ FALLO"
                texto = resultado['texto'] if resultado['texto'] else "NO_DETECTADO"
                f.write(f"{resultado['archivo']:<30} | {texto:<15} | {estado}\n")
        
      

# Función simple para usar directamente
def leer_placas_paddle(carpeta_placas, debug=False):
    """Función simple para leer placas"""
    lector = LectorPlacasPaddle()
    return lector.procesar_carpeta_placas(carpeta_placas, debug)

def leer_placa_individual_paddle(path_imagen, debug=False):
    """Función simple para leer una placa individual"""
    # Para imagen individual, crear carpeta temporal
    carpeta_temp = os.path.join(os.path.dirname(path_imagen), "temp_resultados")
    os.makedirs(carpeta_temp, exist_ok=True)
    
    lector = LectorPlacasPaddle()
    return lector.procesar_imagen_individual(path_imagen, carpeta_temp)