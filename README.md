# IA Projects
Ici sont rassemblés quelques projets et algorithmes d'Intelligence Artificielle, principalement de traitement d'image, que j'ai écrit.

## EfficientNet
EfficientNet est un algorithme de classification décrit dans cette [publication scientifique](https://arxiv.org/pdf/1905.11946.pdf).
Après avoir lu celle-ci, j'ai implémenté l'algorithme en utilisant les technologies suivantes:
 - Python 3.7
 - PyTorch 1.4

> Voici la [DEMO](https://colab.research.google.com/github/lysandrec/IA-Projects/blob/main/EfficientNet/efficientnet-demo.ipynb) de mon code.
> ATTENTION: cette demonstration nécessite un compte Google


Résultats obtenus par mon implémentation de EfficientNet sur le dataset ImageNet:

|Versions        |Pourcentages de classifications réussies|
|----------------|--------------------|
|EfficientNet b0 |76.3%               |
|EfficientNet b1 |78.8%               |
|EfficientNet b2 |79.8%               |
|EfficientNet b3 |81.1%               |
|EfficientNet b4 |82.6%               |
|EfficientNet b5 |83.3%               |
|EfficientNet b6 |84.0%               |
|EfficientNet b7 |84.3%               |