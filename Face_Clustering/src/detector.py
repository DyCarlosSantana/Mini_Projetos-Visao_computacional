import cv2
import os

# 1. Carregar o classificador
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Ler a imagem e VERIFICAR IMEDIATAMENTE
img_path = './assets/raw/F.png'
img = cv2.imread(img_path)

if img is None:
    print(f"❌ ERRO: Não consegui encontrar a imagem em: {os.path.abspath(img_path)}")
    exit() # Interrompe o script aqui se não houver imagem
else:
    print("✅ Imagem carregada com sucesso!")

# 3. Converter para Escala de Cinza
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 4. Detectar os rostos
faces = face_cascade.detectMultiScale(img_gray, 1.1, 4)

# 5. Desenhar o retângulo
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    ''' img: imagem onde desenhar
    (x, y): coordenadas do canto superior esquerdo do retângulo
    (x+w, y+h): coordenadas do canto inferior direito do retângulo
    (255, 0, 0): cor do retângulo (azul em BGR)
    2: espessura da linha do retângulo '''
 
 
# 6. Exibir e Salvar
print(f"Foram encontrados {len(faces)} rostos.")

# Criar a pasta processed se ela não existir (boa prática!)
os.makedirs('./assets/processed', exist_ok=True)

cv2.imwrite('./assets/processed/resultado.jpg', img)
print("✅ Resultado salvo em assets/processed/resultado.jpg")

cv2.imshow('Detector de Rostos', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
