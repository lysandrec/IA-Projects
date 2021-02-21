from math import ceil
import collections
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo

import collections

PATH = {
    'efficientnet-b0': 'https://github.com/lysandrec/Portfolio/releases/download/v1.0/efficientnet-b0.pth',
    'efficientnet-b1': 'https://github.com/lysandrec/Portfolio/releases/download/v1.0/efficientnet-b1.pth',
    'efficientnet-b2': 'https://github.com/lysandrec/Portfolio/releases/download/v1.0/efficientnet-b2.pth',
    'efficientnet-b3': 'https://github.com/lysandrec/Portfolio/releases/download/v1.0/efficientnet-b3.pth',
    'efficientnet-b4': 'https://github.com/lysandrec/Portfolio/releases/download/v1.0/efficientnet-b4.pth',
    'efficientnet-b5': 'https://github.com/lysandrec/Portfolio/releases/download/v1.0/efficientnet-b5.pth',
    'efficientnet-b6': 'https://github.com/lysandrec/Portfolio/releases/download/v1.0/efficientnet-b6.pth',
    'efficientnet-b7': 'https://github.com/lysandrec/Portfolio/releases/download/v1.0/efficientnet-b7.pth',
}

def effnet_coef(version):
    """
    Renvoie les coefficients spécifiques du model (width, depth, res, dropout_rate)
    
    Args:
     - version (str): nom de la version du model
    """
    coefs = {
        # Coefficients:   width,depth,res,dropout_rate
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    }
    return coefs[version]

def effnet_params(width_coefficient, depth_coefficient, image_size, dropout_rate, n_classes=1000, include_top=True):
    """
    Renvoie les paramètres globaux et paramètres des blocks du model (BLOCKS_PARAMS, global_params)
    
    Args:
     - width_coefficient (int)
     - depth_coefficient (int)
     - image_size (int)
     - dropout_rate (float)
     - num_classes (int)
     - include_top (bool)
    """

    # paramètres des blocks du model efficientnet-b0
    # il sera modifié dans la Class EfficientNet pour fit avec les autres versions
    block = collections.namedtuple('BlockArgs', [
        'num_repeat', 'kernel_size', 'stride', 'expand_ratio',
        'input_filters', 'output_filters', 'se_ratio', 'id_skip']
    )
    BLOCKS_PARAMS = [
        block(num_repeat=1, kernel_size=3, stride=[1], expand_ratio=1, input_filters=32, output_filters=16, se_ratio=0.25, id_skip=True),
        block(num_repeat=2, kernel_size=3, stride=[2], expand_ratio=6, input_filters=16, output_filters=24, se_ratio=0.25, id_skip=True),
        block(num_repeat=2, kernel_size=5, stride=[2], expand_ratio=6, input_filters=24, output_filters=40, se_ratio=0.25, id_skip=True),
        block(num_repeat=3, kernel_size=3, stride=[2], expand_ratio=6, input_filters=40, output_filters=80, se_ratio=0.25, id_skip=True),
        block(num_repeat=3, kernel_size=5, stride=[1], expand_ratio=6, input_filters=80, output_filters=112, se_ratio=0.25, id_skip=True),
        block(num_repeat=4, kernel_size=5, stride=[2], expand_ratio=6, input_filters=112, output_filters=192, se_ratio=0.25, id_skip=True),
        block(num_repeat=1, kernel_size=3, stride=[1], expand_ratio=6, input_filters=192, output_filters=320, se_ratio=0.25, id_skip=True),
    ]

    global_params = {
        'width_coefficient': width_coefficient,
        'depth_coefficient': depth_coefficient,
        'image_size': image_size,
        'dropout_rate': dropout_rate,
        'num_classes': n_classes,
        'drop_connect_rate': 0.2,
        'width_divisor': 8,
        'include_top': include_top
    }

    return BLOCKS_PARAMS, global_params

def get_model_params(model_version, n_classes=1000):
    """
    Renvoie les paramètres globaux et paramètres des blocks du model d'une version donnée (blocks_params, global_params)

    Args:
     - model_version (str): nom de la version du model (ex: efficientnet-b3)
    """
    w, d, s, p = effnet_coef(model_version)
    blocks_params, global_params = effnet_params(width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s, n_classes=n_classes)

    return blocks_params, global_params

def calculer_width(in_channels, global_params):
    """
    Calcule le nombre de channels en sortie de la Conv 
    (Calcule la width du model)
    
    Args:
     - in_channels: nombre de channels en entrée
     - global_params: parametres globaux du NN

    Return: nombre de channels en sortie
    """
    multiplicateur = global_params['width_coefficient'] #multiplie par le coef de largeur
    diviseur = global_params['width_divisor']

    in_channels = in_channels * multiplicateur
    out_channels = max(diviseur, int(in_channels + diviseur / 2) // diviseur * diviseur) #formule de l'implementation officielle
    if out_channels < 0.9 * in_channels: # permet de ne pas avoir moins de 90% de la largeur de départ
        out_channels += diviseur
    return int(out_channels)


def calculer_depth(repetition, global_params):
    """
    Calcule le nombre de layers (MBConvBlock)
    (Calcule la depth du model)
    
    Args:
     - repetition: parametre de répétition du block
     - global_params: parametres globaux du NN
    
    Return: nombre de répétition des layers
    """
    multiplicateur = global_params['depth_coefficient']
    return int(ceil(multiplicateur * repetition)) #formule de l'implemeentation officielle


def drop_connect(inputs, prob, training):
    """
    Crée un DropConnect

    Args:
     - inputs (tensor)
     - prob: probabilité déconnection
    retourne les outputs après les connections drop connect
    """
    if not training: #si pas en mode entrainement: ignorer
        return inputs

    batch_size = inputs.shape[0]
    prob_connection = 1 - prob
    #création tensor mask
    random_tensor = prob_connection
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    mask_tensor = torch.floor(random_tensor)

    output = inputs / prob_connection * mask_tensor #application du mask sur les inputs
    return output


def calcule_taille_output_image(taille_image, stride):
    """
    Calcule la taille de l'image de sortie après application de 
    Conv2dSamePadding avec un stride (nécessite static padding donc toute meme taille d'image)

    Args:
     - taille_input_image
     - stride: valeur du stride appliqué
    """
    if taille_image is None: #si pas de taille précisé inutile de calculer
        return None 
    h, w = taille_image if isinstance(taille_image, list) or isinstance(taille_image, tuple) else (taille_image, taille_image)
    stride = stride if isinstance(stride, int) else stride[0]
    h = int(ceil(h / stride))
    w = int(ceil(w / stride))
    return [h, w]


class SwishFn(torch.autograd.Function):
    """swish activation (squelette)"""
    @staticmethod
    def forward(tensor, x):
        resultat = x * torch.sigmoid(x)
        tensor.save_for_backward(x)
        return resultat

    @staticmethod
    def backward(tensor, grad_output):
        x = tensor.saved_tensors[0]
        sigmoid_x = torch.sigmoid(x)
        return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))

class Swish(nn.Module):
    """Fonction d'activation mise en avant par Google Brain ( x*sigmoid(x) )"""
    def forward(self, x):
        return SwishFn.apply(x) #methode de torch.autograd.Function (pour forward et backward)


def get_same_padding_conv2d(image_size=None):
    """
    Choisit padding static si la taille d'image est spécifiée ou padding dynamic sinon
    Renvoie la Conv2D choisi

    Args:
     - image_size (tuple)
    """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size) #classe avec taille_image deja specifiée


class Conv2dDynamicSamePadding(nn.Conv2d):
    """
    Conv2D avec un SAME-padding pour des tailles d'images non spécifiées
    La valeur du padding est calculée dynamiquement puis appliquée dans le 'forward'
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x): #(h:height, w:width) voir les implémentations
        ih, iw = x.size()[-2:]
        sh, sw = self.stride
        kh, kw = self.weight.size()[-2:]
        oh, ow = ceil(ih / sh), ceil(iw / sw) #change la taille en output en accord avec le stride
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]) #applique le padding
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups) #effectue la Conv2D


class Conv2dStaticSamePadding(nn.Conv2d):
    """
    Conv2D avec un SAME-padding pour la tailles des images spécifiée
    La valeur du padding est calculée une fois dans 'init' puis appliquée dans le 'forward'
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        #calcule le padding en fonction de la taille donnée des images 
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        sh, sw = self.stride
        kh, kw = self.weight.size()[-2:]
        oh, ow = ceil(ih / sh), ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity() #si pas de padding à ajouter --> applique fonction qui fait rien

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups) #effectuer la Conv2D
        return x


def get_same_padding_maxPool2d(image_size=None):
    """
    Choisit padding static si la taille d'image est spécifiée ou padding dynamic sinon
    Renvoie le MaxPool choisi

    Args:
     - image_size (tuple)
    """
    if image_size is None:
        return MaxPool2dDynamicSamePadding
    else:
        return partial(MaxPool2dStaticSamePadding, image_size=image_size)


class MaxPool2dDynamicSamePadding(nn.MaxPool2d):
    """
    MawPool2D avec un SAME-padding pour des tailles d'images non spécifiées
    La valeur du padding est calculée dynamiquement puis appliquée dans le 'forward'
    """
    def __init__(self, kernel_size, stride, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.stride = [self.stride] * 2 if isinstance(self.stride, int) else self.stride
        self.kernel_size = [self.kernel_size] * 2 if isinstance(self.kernel_size, int) else self.kernel_size
        self.dilation = [self.dilation] * 2 if isinstance(self.dilation, int) else self.dilation

    def forward(self, x): #Voir les implémentations
        ih, iw = x.size()[-2:]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = ceil(ih / sh), ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding,
                            self.dilation, self.ceil_mode, self.return_indices)

class MaxPool2dStaticSamePadding(nn.MaxPool2d):
    """
    MawPool2D avec un SAME-padding pour la tailles des images spécifiée
    La valeur du padding est calculée une fois dans 'init' puis appliquée dans le 'forward'
    """

    def __init__(self, kernel_size, stride, image_size=None, **kwargs):
        super().__init__(kernel_size, stride, **kwargs)
        self.stride = [self.stride] * 2 if isinstance(self.stride, int) else self.stride
        self.kernel_size = [self.kernel_size] * 2 if isinstance(self.kernel_size, int) else self.kernel_size
        self.dilation = [self.dilation] * 2 if isinstance(self.dilation, int) else self.dilation

        #calcule le padding en fonction de la taille donnée des images 
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = ceil(ih / sh), ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.max_pool2d(x, self.kernel_size, self.stride, self.padding,
                         self.dilation, self.ceil_mode, self.return_indices)
        return x

#NN

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block
    Block tiré de MobileNet v3 (https://arxiv.org/abs/1905.02244)
    """
    def __init__(self, block_params, global_params, image_size=None):
        super().__init__()
        self._block_params = block_params
        #parametres du batch norm
        self._bn_mom = 0.01
        self._bn_eps = 1e-3

        self._se = (self._block_params.se_ratio is not None) and (0 < self._block_params.se_ratio <= 1) #Bool
        self.id_skip = block_params.id_skip  #drop connect ou pas

        # phase d'extansion (Inverted Bottleneck)
        inp = self._block_params.input_filters  #nb de input channels
        output = self._block_params.input_filters * self._block_params.expand_ratio  #nb de output channels (après expension/aggrandissement de la width)
        if self._block_params.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=output, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=output, momentum=self._bn_mom, eps=self._bn_eps)

        # phase de convolution en profondeur
        k = self._block_params.kernel_size
        s = self._block_params.stride
        Conv2d = get_same_padding_conv2d(image_size)
        self._depthwise_conv = Conv2d(in_channels=output, out_channels=output, groups=output, kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=output, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calcule_taille_output_image(image_size, s)

        # Squeeze et Excitation layer
        if self._se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(self._block_params.input_filters * self._block_params.se_ratio)) #!
            self._se_reduce = Conv2d(in_channels=output, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=output, kernel_size=1)

        # phase de convolution finale
        final_output = self._block_params.output_filters
        Conv2d = get_same_padding_conv2d(image_size)
        self._project_conv = Conv2d(in_channels=output, out_channels=final_output, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_output, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = Swish()

    def forward(self, inputs, drop_connect_rate=None):
        """drop_connect_rate (float (entre 0 et 1)"""

        # conv d'extansion (Inverted Bottleneck) et en profondeur
        x = inputs
        if self._block_params.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze et Excitation
        if self._se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x #Swish

        # convolution finale
        x = self._project_conv(x)
        x = self._bn2(x)

        #application du drop connect et des skip connections
        input_filters, output_filters = self._block_params.input_filters, self._block_params.output_filters
        if self.id_skip and self._block_params.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, prob=drop_connect_rate, training=self.training)

            x = x + inputs  # skip connection
        return x


class EfficientNet(nn.Module):
    """EfficientNet model <3"""

    def __init__(self, blocks_params=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_params, list), 'blocks_params doit etre une liste'
        self._global_params = global_params
        self._blocks_params = blocks_params

        #parametres du batch norm
        bn_mom = 0.01
        bn_eps = 1e-3

        #sélection conv static ou dynamique
        image_size = global_params['image_size']
        Conv2d = get_same_padding_conv2d(image_size)

        # stem (tronc)
        in_channels = 3  #rgb
        out_channels = calculer_width(32, self._global_params)  # nb de output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        image_size = calcule_taille_output_image(image_size, 2) #reset de image size

        # MBConvBlock
        self._blocks = nn.ModuleList([])
        for block_params in self._blocks_params:
            # calculer les inputs et outputs filters du block en fonction de ses parametres
            #block_params['input_filters'] = calculer_width(block_params['input_filters'], self._global_params)
            #block_params['output_filters'] = calculer_width(block_params['output_filters'], self._global_params)
            #block_params['num_repeat'] = calculer_depth(block_params['num_repeat'], self._global_params)
            block_params = block_params._replace(
                input_filters=calculer_width(block_params.input_filters, self._global_params),
                output_filters=calculer_width(block_params.output_filters, self._global_params),
                num_repeat = calculer_depth(block_params.num_repeat, self._global_params)
            )

            # 1er block doit faire attention à l'augmentation de stride et de filter size
            self._blocks.append(MBConvBlock(block_params, self._global_params, image_size=image_size))
            image_size = calcule_taille_output_image(image_size, block_params.stride) #re-reset de image size
            if block_params.num_repeat > 1: #modifier block_params pour garder la meme output size à travers les layers
                #block_params['input_filters'] = block_params['output_filters']
                #block_params['stride'] = 1
                block_params = block_params._replace(
                    input_filters=block_params.output_filters,
                    stride=1
                )

            for _ in range(block_params.num_repeat - 1): #car déjà ajouté le 1er
                self._blocks.append(MBConvBlock(block_params, self._global_params, image_size=image_size))

        # Head
        in_channels = block_params.output_filters  # output du block final
        out_channels = calculer_width(1280, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Layer fonctionnel
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params['dropout_rate'])
        self._fc = nn.Linear(out_channels, self._global_params['num_classes'])
        self._swish = Swish()

    def extract_features(self, inputs):
        """
        Utilise layers de convolution pour extraire des features
        Return: l'output de la dernière convolution du model

        Args:
         - inputs (tensor): tensor de l'image
        """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for i, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params['drop_connect_rate']
            if drop_connect_rate:
                drop_connect_rate *= float(i) / len(self._blocks) #repartition du drop connect rate sur l'ensemble des blocks
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """
        (Utilise extract_features pour faire le forward des Convs)
        retourne l'output du model
        """
        # forward des layers de convolution 
        x = self.extract_features(inputs)
        # Pooling et final linear layer
        x = self._avg_pooling(x)
        if self._global_params['include_top']:
            x = x.flatten(start_dim=1)
            x = self._dropout(x)
            x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, version, in_channels=3, n_classes=1000):
        """
        Crée le model efficientnet correspondant à la version donnée

        Args:
         - version (str): nom de la version du model
         - in_channels (int):nombre de channel de l'image d'entrée (rgb --> 3)
        """
        blocks_params, global_params = get_model_params(version, n_classes=n_classes)
        model = cls(blocks_params, global_params)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def from_pretrained(cls, version, in_channels=3, n_classes=1000):
        """
        Crée le model efficientnet correspondant à la version donnée
        et lui load les weights pré entrainé sur ImageNet

        Args:
         - version (str): nom de la version du model
         - in_channels (int):nombre de channel de l'image d'entrée (rgb --> 3)
         - n_classes (int): nombre de classes pour la classification
        """
        model = cls.from_name(version, n_classes=n_classes)
        cls.load_weights(model, PATH[version], fc=(n_classes == 1000))
        model._change_in_channels(in_channels)
        return model

    @staticmethod
    def load_weights(model, path, fc=True):
        """
        load les weights pré-entrainés du fichier du path
        
        Args:
            model: le model efficientnet a qui on charge les weights
            path: path du fichier des weights
            fc: charger les weights des couches fonctionnelles (nn.Linear)
        """
        #state_dict = torch.load(path)
        print(f"Chargement de la sauvegarde du model depuis mon github à {path}")

        state_dict = model_zoo.load_url(path)

        if fc:
            _ = model.load_state_dict(state_dict)
            assert not _.missing_keys, 'KEYS MANQUANTES PENDANT CHARGEMENT DES WEIGHTS INCOMPLET'
        else:
            state_dict.pop('_fc.weight')
            state_dict.pop('_fc.bias')
            _ = model.load_state_dict(state_dict)

        assert not _.unexpected_keys, 'KEYS PAS ATTENDUS PENDANT CHARGEMENT DES WEIGHTS INCOMPLET'

        print(f'Weights chargés avec succès depuis {path}')
    
    def _change_in_channels(self, in_channels):
        """Adjust model's first convolution layer to in_channels, if in_channels not equals 3.
        si in_channels diff de 3, change le 1er layer de convolution pour le fit
        """
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(self._global_params['image_size'])
            out_channels = calculer_width(32, self._global_params)
            self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)