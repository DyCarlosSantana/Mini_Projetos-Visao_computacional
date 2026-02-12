from deepface import DeepFace
import shutil
import os

imagens_path = './assets/raw'
resultados_path = './assets/processed'

img_list = (os.listdir('./assets/raw'))

print(f'Encontrei os seguintes arquivos:')
print(img_list )

identidades_conhecidas = []

for arquivo in img_list:
    if not arquivo.lower().endswith(('.png', '.jpg', '.jpeg')): # Estrutura usada em python para verificar se uma string (geralmente um nome de arquivo) NÃO termina com uma extensão ou sufixo especifico.
        continue

    caminho_completo = os.path.join(imagens_path, arquivo)
    print(f'Processando arquivo: {arquivo}')

    try:
        emb_atual = DeepFace.represent(img_path = caminho_completo, model_name="VGG-Face")[0]["embedding"]
    except Exception as e:
        print(f"Aviso: Não foi possível detectar rosto em {arquivo}. Pulando...")
        continue

    encontrou_id = False

    for index, emb_conhecido in enumerate(identidades_conhecidas):
        distancia = DeepFace.verify(
            img1_path=emb_atual, 
            img2_path=emb_conhecido, 
            model_name="VGG-Face", 
            enforce_detection = False
            )["distance"]
        
        # Ajuda muito a entender por que ele separou as pastas, podemos analisar a distancia 
        print(f"[Debug] Comparando com Pessoa_{index}: Distância = {distancia:.4f}")

        if distancia < 0.65:
            nome_pasta = f"Pessoa_{index}"
            encontrou_id = True
            break

    if not encontrou_id:
        novo_index = len(identidades_conhecidas)
        nome_pasta = f"Pessoa_{novo_index}"
        identidades_conhecidas.append(emb_atual)

    # AGORA: A parte física (Criar pasta e Mover)
    caminho_destino_pasta = os.path.join(resultados_path, nome_pasta)
    os.makedirs(caminho_destino_pasta, exist_ok=True) # Cria a pasta Pessoa_X se não existir

    # Movemos ou copiamos o arquivo da pasta 'raw' para a pasta da pessoa correspondente
    shutil.copy(caminho_completo, os.path.join(caminho_destino_pasta, arquivo))
    print(f"-> {arquivo} movido para {nome_pasta}")

print('Processo Concluido!')