Titulo: Classificação espectral de cepas bacterianas multirresistentes por floresta aleatória aplicada a imagens hiperespectrais

Passo a Passo:
  1º - transformei os arquivos de casca de banana para o csv, para que assim eu consiga trabalhar com os dados e treinar a IA
      Primeiro criei um caminho para conseguir acessar os arquivos de imagem

import numpy as np

# Função SNV aplicada a um conjunto de espectros 2D (n_espectros, n_bandas)
def snv(x):
    return (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)

# Aplica SNV a cada espectro individual da estrutura 3D
# Vamos remodelar para (231*320, 230), aplicar SNV e voltar ao formato original
n_amostras, n_pixels, n_bandas = dados_casca_maca.shape

# Remodela para [total_espectros, bandas]
dados_2d = dados_casca_maca.reshape(-1, n_bandas)  # (231×320, 230)

# Aplica SNV
dados_snv = snv(dados_2d)  # (231×320, 230)

# Volta ao formato original (231, 320, 230)
dados_snv_3d = dados_snv.reshape(n_amostras, n_pixels, n_bandas)