import cv2
import numpy as np
import os

def adjust_contrast_brightness(image, contrast=1.5, brightness=50):
    # Ajuste de contraste e brilho
    image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return image

def resize_and_pad(image, target_size=(640, 480)):
    # Redimensiona a imagem mantendo a proporção e preenche para o tamanho desejado
    old_size = image.shape[:2]
    ratio = min(target_size[0] / old_size[1], target_size[1] / old_size[0])
    new_size = tuple([int(x * ratio) for x in old_size][::-1])

    resized_image = cv2.resize(image, new_size)

    # Cria uma imagem com o tamanho desejado e coloca a imagem redimensionada centralizada
    padded_image = np.full((target_size[1], target_size[0], 3), 0, dtype=np.uint8)
    y_offset = (target_size[1] - new_size[1]) // 2
    x_offset = (target_size[0] - new_size[0]) // 2
    padded_image[y_offset:y_offset + new_size[1], x_offset:x_offset + new_size[0]] = resized_image

    return padded_image

def augment_image(image):
    # Data augmentation: realiza uma única transformação
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Rotaciona a imagem em 15 graus como exemplo de augmentação
    M = cv2.getRotationMatrix2D(center, 15, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    
    return rotated

def preprocess_image(image):
    # Ajuste de contraste e brilho
    image = adjust_contrast_brightness(image, contrast=1.5, brightness=50)
    
    # Redimensiona e padroniza a imagem
    image = resize_and_pad(image, target_size=(640, 480))
    
    # Conversão para escala de cinza (opcional)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Normalização da iluminação (opcional)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Aplicação de Canny para realçar bordas
    edges = cv2.Canny(gray, 50, 150)
    
    # Operações morfológicas para refinar as bordas
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
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
            
            # Pré-processamento da imagem
            processed_image = preprocess_image(image)
            
            # Data augmentation (apenas uma transformação)
            augmented_image = augment_image(processed_image)
            
            # Salva a imagem original pré-processada
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, processed_image)
            print(f'Processed and saved: {output_path}')
            
            # Salva a imagem aumentada
            # aug_output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_aug.png")
            # cv2.imwrite(aug_output_path, augmented_image)
            # print(f'Augmented and saved: {aug_output_path}')

# Caminho da pasta de imagens originais
input_folder = '.\\dataset\\images'
# Caminho para salvar as imagens pré-processadas
output_folder = 'yolov5\\dataset\\images'

# Processa as imagens
process_images(input_folder, output_folder)
