import cv2
from ultralytics import YOLO

# 1. Carrega o modelo otimizado
# Substitua 'best_ncnn_model' pelo caminho exato da pasta ou arquivo gerado na sua Opção 1
# Se foi NCNN, geralmente é uma pasta chamada 'best_ncnn_model' ou 'best_web_model'
model = YOLO("best_ncnn_model", task="segment")

# 2. Inicializa a webcam
# O argumento 0 abre a webcam padrão (USB ou câmera do Pi). Se tiver mais de uma, tente 1, 2, etc.
cap = cv2.VideoCapture(0)

# Define a resolução de captura da webcam (opcional, mas ajuda a não sobrecarregar o barramento USB)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Erro: Não foi possível abrir a webcam.")
    exit()

print("Sistemas iniciados. Pressione 'q' na janela do vídeo para sair.")

# Loop de captura e processamento
while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao receber o frame da webcam. Encerrando...")
        break

    # 3. Executa a inferência do YOLO
    # imgsz=320 mantém a resolução baixa que garantiu a performance
    # stream=True impede o acúmulo de frames na memória RAM
    results = model(frame, imgsz=320, stream=True)

    # 4. Processa os resultados
    for r in results:
        # O método .plot() renderiza as caixas, máscaras de segmentação e labels automaticamente
        annotated_frame = r.plot()
        
        # 5. Exibe o frame com as viaturas identificadas
        cv2.imshow("Detector de Viaturas EB - Raspberry Pi 4", annotated_frame)

    # Condição de parada: pressionar a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpeza e fechamento de janelas
cap.release()
cv2.destroyAllWindows()
