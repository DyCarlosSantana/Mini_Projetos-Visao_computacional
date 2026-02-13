from deepface import DeepFace
from datetime import datetime
import shutil
import os

# --- CONFIGURAÇÕES ---
imagens_path = './assets/raw'
resultados_path = './assets/processed'
arquivo_relatorio = os.path.join(resultados_path, "relatorio_final.txt")

# Limiar de decisão: Quanto maior, mais flexível (aceita mais variações)
# Quanto menor, mais rigoroso (separa a mesma pessoa se a luz mudar)
LIMIT_THRESHOLD = 0.65 

# Garante que a pasta de resultados existe
os.makedirs(resultados_path, exist_ok=True)

# Lista os arquivos da pasta original
img_list = os.listdir(imagens_path)

print(f'--- INICIANDO ORGANIZAÇÃO ---')
print(f'Arquivos encontrados: {len(img_list)}')

# MEMÓRIA DO SISTEMA
identidades_conhecidas = []  # Guarda os vetores (embeddings) de quem já vimos
stats = {}                   # Guarda a contagem de fotos (ex: {'Pessoa_0': 5})

# --- LOOP PRINCIPAL ---
for arquivo in img_list:
    # 1. Filtro de Segurança (Ignora arquivos que não são imagem)
    if not arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    caminho_completo = os.path.join(imagens_path, arquivo)
    print(f'\nProcessando: {arquivo}')

    # 2. Extração do Rosto (Try/Except para não travar em fotos ruins)
    try:
        # Pega o primeiro rosto encontrado e seu vetor numérico
        emb_atual = DeepFace.represent(img_path=caminho_completo, model_name="VGG-Face")[0]["embedding"]
    except:
        print(f"⚠️ Aviso: Rosto não detectado em {arquivo}. Pulando.")
        continue

    # 3. Comparação e Decisão (Um contra Todos)
    encontrou_id = False
    nome_pasta = ""

    for index, emb_conhecido in enumerate(identidades_conhecidas):
        # Compara os vetores usando a métrica COSSENO
        try:
            resultado = DeepFace.verify(
                img1_path = emb_atual, 
                img2_path = emb_conhecido, 
                model_name = "VGG-Face", 
                distance_metric = "cosine",
                enforce_detection = False
            )
            distancia = resultado["distance"]
        except:
            continue

        # Se a distância for pequena, é a mesma pessoa
        if distancia < LIMIT_THRESHOLD:
            nome_pasta = f"Pessoa_{index}"
            encontrou_id = True
            print(f"   >>> MATCH! Pertence à {nome_pasta} (Dist: {distancia:.4f})")
            break # Pare de procurar

    # 4. Aprendizado (Se ninguém foi encontrado)
    if not encontrou_id:
        novo_index = len(identidades_conhecidas)
        nome_pasta = f"Pessoa_{novo_index}"
        identidades_conhecidas.append(emb_atual) # Adiciona nova face à memória
        print(f"   >>> NOVA IDENTIDADE! Criando {nome_pasta}")

    # 5. Ação Física (Mover/Copiar e Contar)
    caminho_destino_pasta = os.path.join(resultados_path, nome_pasta)
    os.makedirs(caminho_destino_pasta, exist_ok=True)

    # Copia o arquivo
    shutil.copy(caminho_completo, os.path.join(caminho_destino_pasta, arquivo))
    
    # Atualiza o contador para o relatório
    # "Busque a pessoa X. Se não existir, devolva 0. Depois some 1."
    stats[nome_pasta] = stats.get(nome_pasta, 0) + 1

# --- FIM DO PROCESSAMENTO ---

# 6. Geração do Relatório TXT
print("\n📝 Gerando relatório final...")

with open(arquivo_relatorio, "w", encoding="utf-8") as f:
    f.write("=== RELATÓRIO DE ORGANIZAÇÃO ===\n")
    f.write(f"Data de Execução: {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}\n")
    f.write("================================\n\n")
    
    f.write(f"Total de Identidades Únicas: {len(identidades_conhecidas)}\n")
    f.write(f"Total de Imagens Processadas: {sum(stats.values())}\n\n")
    
    f.write("--- Detalhe por Pasta ---\n")
    for pessoa, quantidade in stats.items():
        f.write(f"- {pessoa}: {quantidade} imagens\n")

print(f"✅ Sucesso Total! Verifique a pasta '{resultados_path}'.")