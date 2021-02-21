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
    'input_filters', 'output_filters', 'se_ratio', 'id_skip'])
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