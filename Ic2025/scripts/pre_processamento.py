import leitura_normalizacao
from chemotools.scatter import StandardNormalVariate


snv = StandardNormalVariate()
spectra_snv = snv.fit_transform(espectros_matriz_norm)