# -*- coding: utf-8 -*-
"""
Autor: André Pacheco
Email: pacheco.comp@gmail.com

"""

import torch
from torch import nn
from torchvision import transforms, models, datasets

class Net (nn.Module):
    """
    This class get a resnet model, changes its FC layer and include the extra information on it (if applicable)
    """

    def __init__(self, resnet, num_class, freeze_conv=False, n_extra_info=0, p_dropout=0.5, neurons_class=256,
                 feat_reducer=None, classifier=None):
        """

        :param resnet: the resnet model from torchvision.model. Ex: model.resnet50
        :param num_class: the number of task's classes
        :param freeze_conv: if you'd like to freeze the extraction map from the model
        :param n_extra_info: the number of extra information you wanna include in the model
        :param p_dropout: the dropout probability
        """
        super(Net, self).__init__()

        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # freezing the convolution layers
        if freeze_conv:
            for param in self.features.parameters():
                param.requires_grad = False

        # Feature reducer
        if feat_reducer is None:
            self.feat_reducer = nn.Sequential(
                nn.Linear(2048, neurons_class),
                nn.BatchNorm1d(neurons_class),
                nn.ReLU(),
                nn.Dropout(p=p_dropout)
            )
        else:
            self.feat_reducer = feat_reducer

        # Here comes the extra information (if applicable)
        if classifier is None:
            self.classifier = nn.Linear(neurons_class + n_extra_info, num_class)
        else:
            self.classifier = classifier



    def forward(self, img, extra_info=None):

        x = self.features(img)
        

        # Flatting
        x = x.view(x.size(0), -1)

        x = self.feat_reducer(x)

        if extra_info is not None:
            agg = torch.cat((x, extra_info), dim=1)
        else:
            agg = x

        res = self.classifier(agg)

        return res

    def feature_list(self, img, extra_info=None):
        # https://discuss.pytorch.org/t/accessing-intermediate-layers-of-a-pretrained-network-forward/12113
        # using the above reference, built this function to return intermediate features
        # the other reference was the original function here: https://github.com/pokaxpoka/deep_Mahalanobis_detector/blob/90c2105e78c6f76a2801fc4c1cb1b84f4ff9af63/models/resnet.py
        # the feature_list function is essentially the forward function, returns intermediate features in `out_list` 
        # instead of full forward pass results  
        out = img
        out_list = []
 

        for ii, layer in enumerate(self.features):
            print("COUNT-A", ii, type(layer))
            out = layer(out)
            if isinstance(layer, nn.Sequential) or isinstance(layer, nn.ReLU):
                out_list.append(out)  
                
        out = out.view(out.size(0), -1)
        out = self.feat_reducer(out)
        if extra_info is not None:
            agg = torch.cat((out, extra_info), dim=1)
        else:
            agg = out
        res = self.classifier(agg) 
        return res, out_list

    # function to extract a specific feature
    def intermediate_forward(self, img, layer_index):
        # implemented this one myself as well.. See the notes in the feature_list function
        out = img
        count = 0
        for ii, layer in enumerate(self.features): 
            out = layer(out)
            if isinstance(layer, nn.ReLU) and layer_index == 0: return out 
            elif isinstance(layer, nn.Sequential):
                count += 1
                if count == layer_index: return out     