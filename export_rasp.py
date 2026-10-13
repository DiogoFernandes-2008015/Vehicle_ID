from ultralytics import YOLO

model = YOLO("runs/segment/runs/segment/viaturas_eb-3/weights/best.pt")
# Exportando para NCNN (geralmente o melhor formato para CPU ARM do Pi)
model.export(format="ncnn", imgsz=320)