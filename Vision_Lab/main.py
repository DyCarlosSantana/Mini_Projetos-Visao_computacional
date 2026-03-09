import cv2     
# OpenCV - Para processamento de imagens
import numpy as np  
# NumPy - Para trabalhar com arrays de forma eficiente caso necessario
import matplotlib.pyplot as plt  
# Matplotlib - Para exibir imagens na tela
# import 'nome da biblioteca' as 'apelido'

IMG_PATH = 'assets/image.png'


def carregar_imagem(caminho):
    """
    Carrega do disco.
    Args:
        caminho (str): Caminho do arquivo de imagem
    Returns:
        numpy.ndarray: Imagem carregada em fomato BGR ou None se o carregamento falhar.
    """
    img = cv2.imread(caminho)
    if img is None:
        print(f"Erro: Não foi possivel carregar {caminho}")
        return None
    
    print(f"Imagem Carregada: {img.shape}") # .shape para mostrar as dimensões da Imagem em uma tupla (altura, largura, canais)
    return img


def exibir_imagem_original(imagem):
    """
    Exibir uma imagem em formato BGR
    Arg:
        imagem: (numpy.ndarray): Imagem em formato BGR (opencv)
    """
    if imagem is None:
        print("Imagem Invalida! Nada a exibir")
        return
    
    img_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    print("Exibindo imagem...")
    plt.imshow(img_rgb)
    plt.axis('off')  # Remove os eixos (números) da imagem
    plt.title('Imagem Original - RGB')  # Adiciona título
    plt.show()

# ---- Aplicação dos Filtros ----

def aplicar_cinza(imagem):
    """
    Convete a imagem para escala de cinza e exibe
    Arg:
        imagem(numpy.ndarray): Imagem em formato BGR
    """
    if imagem is None:
        print("Imagem Invalida! Nada a exibir")
        return None
    
    img_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    print("Escalas de cinza aplicadas")
    return img_cinza

def aplicar_blur(imagem):
    """
    Aplicar filtro Blur (desfoque) na imagem.
    Arg:
        imagem (numpy.ndarray): Imagem em BGR ou escala de cinza
    Return: 
        numpy.ndarray: Imagem com blur aplicado ou None se falhar
    """
    if imagem is None:
        print("Imagem Invalida! Nada a exibir")
        return None
    
    img_blur = cv2.GaussianBlur(imagem, (15, 15), 0)
    print("Blur aplicado - Kernel: (5, 5), 0")
    return img_blur

def detectar_bordas(imagem):
    """
    Detecta os contornos/bordas da imagem.
    Arg:
        imagem (numpy.ndarray): Imagem em escala de cinza
    Return: 
        numpy.ndarray: Imagem com bordas detectadas ou None se falhar
    """
    if imagem is None:
        print("Imagem Invalida! Nada a exibir")
        return None
    if len(imagem.shape) == 3: # imagem.shape retorna a tupla com 3 valores se for colorida, indicado que ainda não está em escalas de cinza e precisa ser convertida
        img_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    else:
        img_cinza = imagem

    img_bordas = cv2.Canny(img_cinza, 100, 200)
    print("Bordas detectadas - Thresholdes: (100, 200)")
    return img_bordas

def aplicar_threshold(imagem):
    """
    Aplica threshold (binarização) usando método Otsu.
    Args:
        imagem (numpy.ndarray): Imagem em BGR ou escala de cinza
    Returns:
        numpy.ndarray: Imagem binarizada (preto/branco) ou None se falhar
    """
    if imagem is None:
        print("Imagem inválida! Nada a exibir")
        return None
    
    if len(imagem.shape) == 3:
        img_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    else:
        img_cinza = imagem
    
    retval, img_thresh = cv2.threshold(
        img_cinza, 
        0,        # Threshold (ignorado com Otsu)
        255,      # Valor máximo (branco)
        cv2.THRESH_BINARY + cv2.THRESH_OTSU  # Tipo: binário + Otsu
    )
    print(f"Threshold aplicado - Valor Otsu: {retval:.2f}")
    return  img_thresh

# Função para exibir o grid de comparação
def exibir_grid_comparativo(original, cinza, blur, bordas, threshold):
    """
    Exibe todas as transformações em um grid de comparação 2x3.
    Args:
        original (numpy.ndarray): Imagem original em BGR
        cinza (numpy.ndarray): Imagem em escala de cinza
        blur (numpy.ndarray): Imagem com blur aplicado
        bordas (numpy.ndarray): Imagem com bordas detectadas
        threshold (numpy.ndarray): Imagem binarizada
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    # Criar figura com grid 2x3 (linhas x Colunas)
    # figsize=(15, 10) = largura 15", altura 10" (polegadas)

    #conveerte a imagem original BGR -> RGB para exibição correta
    original_rgb = cv2. cvtColor(original, cv2.COLOR_BGR2RGB)
    #[0, 0] Posição da Imagem original
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title('Original', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    #[0, 1] Posição da imagem em escalas de cinza
    axes[0, 1].imshow(cinza, cmap='gray')
    axes[0, 1].set_title('Escala de Cinza', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    #[0, 2] Posição da Imagem com Gaussian Blur
    if len(blur.shape) == 3:
        blur_rgb = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
        axes[0, 2].imshow(blur_rgb)
    else:
        axes[0, 2].imshow(blur, cmap='gray')
    axes[0, 2].set_title('Gaussian Blur', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')

    # [1, 0] Posição da Detecção de Bordas (Canny)
    axes[1, 0].imshow(bordas, cmap='gray')
    axes[1, 0].set_title('Bordas (Canny)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # [1, 1] Posição de Threshold (Binarização)
    axes[1, 1].imshow(threshold, cmap='gray')
    axes[1, 1].set_title('Threshold (Otsu)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    # [1, 2] Deixar vazio (mas com layout simetrico)
    axes[1, 2].axis('off')

    # Ajusta espaçamento entre subplots
    plt.tight_layout()

    # Exibir janela
    plt.show()


    print("Grid de Comparação exibido")
if __name__ == "__main__":
    print("Iniciando Vision Lab...")

    imagem = carregar_imagem(IMG_PATH)
    if imagem is None:
        print("Não foi possível processar a imagem. Encerrando!")
    else:
        print(" --- Processando filtros ---")
        img_cinza = aplicar_cinza(imagem)
        img_blur = aplicar_blur(imagem)
        img_bordas = detectar_bordas(imagem)
        img_thresh = aplicar_threshold(imagem)

        print("\nTodos os filtros aplicados com sucesso!")
        print("Exibindo Grid de Comparação")
        exibir_grid_comparativo(imagem, img_cinza, img_blur, img_bordas, img_thresh)

        print("Vision Lab v2.5")
