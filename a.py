import os

labels_dir = 'yolov5\\dataset\\labels'  # Certifique-se de que esse diretório está correto

for filename in os.listdir(labels_dir):
    if filename.endswith('.txt'):
        filepath = os.path.join(labels_dir, filename)
        with open(filepath, 'r') as file:
            lines = file.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                parts[0] = '0'  # Alterar para a única classe disponível
                new_lines.append(' '.join(parts))
        
        with open(filepath, 'w') as file:
            file.write('\n'.join(new_lines))
