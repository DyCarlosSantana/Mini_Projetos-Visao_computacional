from deepface import DeepFace # Biblioteca para reconhecimento facial
import os

# 1. Definimos o caminho da imagem que queremos processar
img_path = './assets/raw/F.png'

# Verificação de segurança (como aprendemos no Marco 01)
if not os.path.exists(img_path):
    print(f"Erro: Arquivo {img_path} não encontrado!")
    exit()

print("Iniciando a extração da identidade digital (isso pode demorar na primeira vez)...")

# 2. O comando principal: represent()
# Ele faz 3 coisas: detecta o rosto, alinha ele e gera os números (embeddings)
resultados = DeepFace.represent(img_path = img_path, model_name = "VGG-Face")

# 3. Como a DeepFace pode achar vários rostos, ela retorna uma LISTA de dicionários.
# Vamos pegar o primeiro rosto encontrado:
primeiro_rosto = resultados[0]
assinatura_digital = primeiro_rosto["embedding"]

# 4. Vamos ver o resultado!
print(f"\n✅ Sucesso! O rosto foi transformado em um vetor de {len(assinatura_digital)} números.")
print(f"Os 5 primeiros números da 'identidade' são: {assinatura_digital[:5]}")
print(f"Nivel de confiança: {primeiro_rosto['face_confidence']*100:.2f}%")
 