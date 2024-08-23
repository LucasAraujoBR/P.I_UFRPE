# %% [markdown]
# # Pré-Processamento das Imagens

# %%
import cv2
import numpy as np
import os

def preprocess_image(image):
    # Redimensiona a imagem se necessário
    image = cv2.resize(image, (640, 480))
    
    # Conversão para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Normalização da iluminação com CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Remoção de ruído com GaussianBlur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Aplicação de Canny para realçar bordas
    edges = cv2.Canny(blurred, 50, 150)
    
    # Aplica a segmentação de cor para destacar veículos
    lower_bound = np.array([0, 0, 120])  # Exemplo para destacar tons mais escuros ou avermelhados (ajuste conforme necessário)
    upper_bound = np.array([255, 255, 255])
    mask = cv2.inRange(image, lower_bound, upper_bound)
    
    # Combina as bordas com a máscara de segmentação
    combined = cv2.bitwise_and(edges, edges, mask=mask)
    
    # Operações morfológicas para refinar a imagem
    kernel = np.ones((5,5), np.uint8)
    morphed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    return morphed

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".PNG") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read the image {filename}. Skipping.")
                continue
            
            processed_image = preprocess_image(image)
            
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, processed_image)
            print(f'Processed and saved: {output_path}')

# Caminho da pasta de imagens originais
input_folder = 'dataset/images'
# Caminho para salvar as imagens pré-processadas
output_folder = 'dataset/processed_images'

# Processa as imagens
process_images(input_folder, output_folder)

# %% [markdown]
# # Integração com YoloV10

# %%
import torch
from yolov10.models.common import DetectMultiBackend
import cv2

def detect_vehicles(video_path):
    model = DetectMultiBackend(weights='path_to_yolov10_weights.pth')
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        results.show()  # Exibe as detecções
    
    cap.release()

# Vídeo pré-processado
processed_video_path = 'path_to_save_processed_video.avi'

# Realiza a detecção no vídeo pré-processado
detect_vehicles(processed_video_path)


