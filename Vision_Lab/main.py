import cv2     # OpenCV - Para processamento de imagens
import numpy as np  # NumPy - Para trabalhar com arrays de forma eficiente
import matplotlib.pyplot as plt  # Matplotlib - Para exibir imagens na tela
# import 'nome da biblioteca' as 'apelido'

IMG_PATH = 'assets/foto.jpg'


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
        print("⚠️ Imagem Invalida! Nada a exibir")
        return
    
    img_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')  # Remove os eixos (números) da imagem
    plt.title('Imagem Original - RGB')  # Adiciona título
    plt.show()

def converter_exibir_cinza(imagem):
    """
    Convete a imagem para escala de cinza e exibe
    Arg:
        imagem(numpy.ndarray): Imagem em formato BGR
    """
    if imagem is None:
        print("⚠️ Imagem Invalida! Nada a exibir")
        return
    
    img_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    plt.imshow(img_cinza, cmap='gray')
    plt.axis('off')
    plt.title('Imagem em Escala de Cinza')
    plt.show()

if __name__ == "__main__":
    print("Iniciando Vision Lab...")

    imagem = carregar_imagem(IMG_PATH)
    if imagem is None:
        print("⚠️ Não foi possível processar a imagem. Encerrando.")
    else:
        print("Exibindo imagem original...")
        exibir_imagem_original(imagem)
        
        print("Convertendo para escala de cinza...")
        converter_exibir_cinza(imagem)
        
        print("✅ Processamento concluído!")