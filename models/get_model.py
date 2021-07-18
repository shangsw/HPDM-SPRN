# -*- coding: utf-8 -*-
from . import DFFN, DHCNet, SSRN, FDSSC, SPRN, BASSNet

# get models by name
def get_model(name, dataset, n_classes, n_bands, patch_size):
    if name == 'BASSNet':
        model = BASSNet.BASSNet(dataset, n_bands, n_classes, patch_size)
    elif name == 'SSRN':
        model = SSRN.SSRN(n_bands, n_classes)
    elif name == 'FDSSC':
        model = FDSSC.FDSSC(n_bands, n_classes)
    elif name == 'DHCNet':
        model = DHCNet.DHCNet(n_bands, n_classes)
    elif name == 'DFFN':
        model = DFFN.DFFN(dataset, n_bands, n_classes)
    elif name == 'SPRN':
        model = SPRN.SPRN(dataset, n_bands, n_classes)

    elif name == 'HPDM-BASSNet':
        model = BASSNet.HPDM_BASSNet(dataset, n_bands, n_classes, patch_size)
    elif name == 'HPDM-SSRN':
        model = SSRN.HPDM_SSRN(n_bands, n_classes)
    elif name == 'HPDM-FDSSC':
        model = FDSSC.HPDM_FDSSC(n_bands, n_classes)
    elif name == 'HPDM-DFFN':
        model = DFFN.HPDM_DFFN(dataset, n_bands, n_classes)
    elif name == 'HPDM-DHCNet':
        model = DHCNet.HPDM_DHCNet(n_bands, n_classes)
    elif name == 'HPDM-SPRN':
        model = SPRN.HPDM_SPRN(dataset, n_bands, n_classes)
    else:
        raise KeyError("{} model is unknown.".format(name))

    return model