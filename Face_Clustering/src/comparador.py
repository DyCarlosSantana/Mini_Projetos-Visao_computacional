from deepface import DeepFace
import os

# Caminhos das imagens a serem comparadas
# Rosto 1
img1_path = './assets/raw/D.jpg'
img2_path = './assets/raw/H.jpg'
img3_path = './assets/raw/F.png'
# Rosto 2 - diferente para comparação negativa
img4_path = './assets/raw/E.jpg'

# Verificação de segurança usando um loop para evitar repetição de código
for path in [img1_path, img2_path, img3_path, img4_path]:
    if not os.path.exists(path):
        print(f"Erro: Arquivo {path} não encontrado!")
        exit()

print("Iniciando a comparação dos rostos...")
# Comparar os rostos usando a função verify() da DeepFace
comparacao1 = DeepFace.verify(img1_path, img2_path, model_name= "VGG-Face")
comparacao2 = DeepFace.verify(img1_path, img3_path, model_name= "VGG-Face")
comparacao3 = DeepFace.verify(img1_path, img4_path, model_name= "VGG-Face")

print("\nResultados da comparação:")
# Exibir os resultados
# A função verify() retorna um dicionário com várias informações, mas vamos focar em "verified" (booleano) e "distance" (número que indica a distância entre os vetores de características dos rostos)
print(f"Comparação entre {os.path.basename(img1_path)} e {os.path.basename(img2_path)}: {'Mesma pessoa' if comparacao1['verified'] else 'Pessoas diferentes'} (Confiança: {comparacao1['distance']:.4f})")
print(f"Comparação entre {os.path.basename(img1_path)} e {os.path.basename(img3_path)}: {'Mesma pessoa' if comparacao2['verified'] else 'Pessoas diferentes'} (Confiança: {comparacao2['distance']:.4f})")
print(f"Comparação entre {os.path.basename(img1_path)} e {os.path.basename(img4_path)}: {'Mesma pessoa' if comparacao3['verified'] else 'Pessoas diferentes'} (Confiança: {comparacao3['distance']:.4f})")
# os.path.basename() é usado para extrair apenas o nome do arquivo da imagem 
