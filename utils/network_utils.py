from network.resnet import * 

def get_backbone(trunk):
    if trunk == 'resnet18':
        backbone = resnet18()
        backbone_channels = 512
    elif trunk == 'resnet34':
        backbone = resnet34()
        backbone_channels = 512
    elif trunk == 'resnet50':
        backbone = resnet50()
        backbone_channels = 2048 
    elif trunk == 'resnet101':
        backbone = resnet50()
        backbone_channels = 2048
    else:
        raise NotImplementedError('Backbone "{}" not currently supported'.format(trunk))
    return backbone, backbone_channels

def forgiving_load_state_dict(state_dict, model, logger):
    keys_dict = state_dict.keys()
    keys_model = model.state_dict().keys()

    keys_shared = [k for k in keys_dict if k in keys_model]
    keys_missing = [k for k in keys_model if k not in keys_dict]
    keys_unexpected = [k for k in keys_dict if k not in keys_model]

    load_dict = {k:v for k,v in state_dict.items() if k in keys_shared}
    model.load_state_dict(load_dict, strict = False)

    logger.info('Missing Keys in checkpoint : ')
    for k in keys_missing:
        logger.info('\t\t{}'.format(k))
    logger.info('Unexpected Keys in checkpoint : ')
    for k in keys_unexpected:
        logger.info('\t\t{}'.format(k))
