from detector import procesar_video
from detector.lector_placas import leer_placas_paddle

if __name__ == "__main__":
    procesar_video(
        video_path="data/video.mp4",
        modelo_vehiculos_path='yolo11n.pt',
        # modelo_vehiculos_path='yolov8n.pt',
        modelo_placas_path="models/best.pt"
    )
    
    print("\nLeyendo placas con PaddleOCR...")
    leer_placas_paddle("placas/", debug=True)
