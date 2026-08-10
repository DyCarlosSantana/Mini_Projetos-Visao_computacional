# 👁️ Mini Projetos - Visão Computacional

Repositório de estudo prático em visão computacional com Python — dois mini-projetos independentes, cada um explorando uma camada diferente da área: processamento clássico de imagem (`Vision_Lab`) e reconhecimento/agrupamento facial com deep learning (`Face_Clustering`).

---

## Sobre o repositório

Diferente do `Cleitinho_ChatBot` (outro repositorio com um mini-projeto de estudo prático, que segue uma trilha linear), aqui os dois projetos são **independentes entre si** — não há progressão direta de um para o outro, mas há uma progressão conceitual implícita:

- **`Vision_Lab`** cobre processamento de imagem clássico: OpenCV puro, sem machine learning. Espaço de cor, filtros, detecção de bordas, binarização.
- **`Face_Clustering`** vai além: usa um classificador Haar Cascade para detecção de rosto e, na sequência, uma rede neural pré-treinada (via DeepFace) para extrair *embeddings* faciais e agrupar pessoas automaticamente por similaridade — sem saber previamente quantas pessoas existem no conjunto de fotos.

---

## `Vision_Lab`

Laboratório de filtros clássicos de processamento de imagem. Roda em cima de uma única imagem (`assets/image.png`) e aplica, em sequência: escala de cinza, blur gaussiano, detecção de bordas (Canny) e binarização (threshold de Otsu) — exibindo tudo num grid comparativo.

<p align="center">
   <img width="75%" alt="grid_comparativo" src="https://github.com/user-attachments/assets/3b6983a9-2e33-4032-ac56-69a1311eccd3" />
</p>

> Print gerado executando o pipeline real do `main.py` sobre a imagem de teste do próprio repositório.

### Pipeline

| Etapa | Função | O que faz |
|---|---|---|
| 1 | `carregar_imagem()` | Lê o arquivo do disco via `cv2.imread`, valida se carregou |
| 2 | `aplicar_cinza()` | Converte BGR → escala de cinza |
| 3 | `aplicar_blur()` | Aplica `GaussianBlur` com kernel 15×15 |
| 4 | `detectar_bordas()` | `cv2.Canny` com thresholds 100/200 sobre a versão em cinza |
| 5 | `aplicar_threshold()` | Binarização automática via método de Otsu |
| 6 | `exibir_grid_comparativo()` | Monta um grid 2×3 com `matplotlib` mostrando todas as etapas lado a lado |

Um detalhe que vale registrar: o OpenCV lê e trabalha imagens em **BGR**, não RGB — por isso toda função que exibe a imagem original ou colorida faz a conversão `cv2.COLOR_BGR2RGB` antes de passar pro `matplotlib` (que espera RGB). Ignorar essa conversão é o erro clássico — as cores saem trocadas (azul vira vermelho e vice-versa).

Existe inclusive um segundo arquivo no projeto, `teste_sem_conversao.py`, que  feito propositalmente pra demonstrar esse erro (o nome do arquivo já entrega — "teste sem conversão"). Vale como material de estudo do próprio conceito.

---

## `Face_Clustering`

Pipeline de três estágios que recebe uma pasta de fotos desorganizadas e devolve subpastas separadas por pessoa — sem que o número de pessoas seja informado com antecedência.

### Pipeline

1. **Detecção (`detector.py`)** — usa **Haar Cascade**, um classificador clássico (não é deep learning) treinado sobre padrões de contraste típicos de um rosto (ex: a ponte do nariz costuma ser mais clara que a região dos olhos). É rápido porque descarta candidatos cedo: se o primeiro filtro não encontra nada, a "cascata" para ali e não tenta os filtros seguintes.

2. **Codificação (`encoder.py`)** — usa **DeepFace** (modelo `VGG-Face`) para transformar um rosto detectado num vetor numérico (*embedding*) de alta dimensão — a "identidade digital" do rosto.

3. **Clustering incremental (`cluster.py`)** — para cada nova imagem, compara o embedding atual contra todos os embeddings já vistos usando distância de cosseno. Se a distância for menor que `LIMIT_THRESHOLD` (0.65), a imagem entra na pasta da pessoa já conhecida; caso contrário, gera uma nova identidade. É uma abordagem *online* (um contra todos, greedy) — não é k-means nem clustering hierárquico clássico, é mais simples e mais rápido, mas sensível à ordem de processamento e ao valor do threshold.

O script gera automaticamente um relatório (`assets/processed/relatorio_final.txt`). Rodando sobre o conjunto de teste do repositório, o resultado real foi:

```
Total de Identidades Encontradas: 9
Total de Imagens Processadas: 15
- Pessoa_2: 5 imagens
- Pessoa_3: 2 imagens
- Pessoa_6: 2 imagens
- (demais: 1 imagem cada)
```

Achar 9 identidades pra um conjunto onde a maioria das pastas tem só 1 foto é sinal de threshold conservador demais — o algoritmo está separando fotos que provavelmente são da mesma pessoa em identidades diferentes (falso negativo), o que é mais seguro que o contrário, mas indica que `0.65` pode estar abaixo do ideal para esse conjunto de imagens. Ajustar esse valor é o primeiro experimento óbvio a fazer.

Existe também `comparador.py`, um script separado e mais simples — compara pares específicos de imagens diretamente com `DeepFace.verify()`, sem clustering. Bom para testar rapidamente se duas fotos são da mesma pessoa antes de rodar o pipeline completo.

---

## Estrutura de pastas

```
Mini_Projetos-Visao_computacional/
├── Vision_Lab/
│   ├── main.py                  # Pipeline completo de filtros
│   ├── teste_sem_conversao.py   # Demonstração do erro BGR/RGB
│   ├── Anotações.txt
│   └── assets/
│       └── image.png
└── Face_Clustering/
    ├── main.py                  # Vazio (ver §5, ponto 3)
    ├── src/
    │   ├── detector.py          # Marco 01 — detecção (Haar Cascade)
    │   ├── encoder.py           # Marco 02 — embeddings (DeepFace)
    │   ├── cluster.py           # Marco 03/04 — clustering + relatório
    │   └── comparador.py        # Comparação direta entre pares de imagens
    ├── Anotações.txt
    └── assets/
        ├── raw/                 # Fotos originais
        └── processed/           # Saída organizada por pessoa + relatório
```

---

## Stack técnica

| Biblioteca | Uso |
|---|---|
| OpenCV (`opencv-contrib-python`) | Leitura, transformação e filtros de imagem |
| NumPy | Representação matricial das imagens |
| Matplotlib | Exibição dos resultados (grids, imagens) |
| DeepFace | Extração de embeddings faciais e comparação (`verify`) |
| scikit-learn | Citado nas anotações para clustering — não usado ainda no `cluster.py` atual (a lógica implementada é uma comparação incremental manual, não `KMeans`/`DBSCAN`) |

---

## Como executar

```bash
# Vision_Lab
cd Vision_Lab
python main.py

# Face_Clustering — os módulos rodam individualmente por enquanto
cd Face_Clustering/src
python detector.py       # Marco 01: detecção isolada
python encoder.py        # Marco 02: extração de embedding de uma imagem
python cluster.py        # Marco 03/04: pipeline completo de agrupamento
python comparador.py     # Comparação direta entre pares específicos
```

---

## Observações

- O dataset de imagens usadas para testes incluia imagem de pessoa reais (artistias still de produção audiovisual). Antes de tornar o repositorio público retirei todos as imagens de (`assets`) usadas para os testes. Mas basta incluir novas imagens e realizar seus proprios teste com os scripts.
- Pode existir a necessidade de depuração no scripts principalmente revendo os (`paths`) após os arquivos de imagem terem sido retirados.
