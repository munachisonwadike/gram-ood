# -*- coding: utf-8 -*-
"""
Autor: André Pacheco
Email: pacheco.comp@gmail.com
"""

import torch
from torch import nn


class Net (nn.Module):

    def __init__(self, vgg, num_class, freeze_conv=False, n_extra_info=0, p_dropout=0.5, neurons_class=256,
                 feat_reducer=None, classifier=None):

        super(Net, self).__init__()

        self.features = nn.Sequential(*list(vgg.children())[:-1])

        # freezing the convolution layers
        if freeze_conv:
            for param in self.features.parameters():
                param.requires_grad = False

        # Feature reducer
        if feat_reducer is None:
            self.feat_reducer = nn.Sequential(
                nn.Linear(25088, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(p=p_dropout),
                nn.Linear(1024, neurons_class),
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
        for ii, layer in enumerate(*self.features[:-1]): #this just exclused the AdaptiveAvgPool2d
            # print(ii, type(layer))
            out = layer(out)
            if isinstance(layer, nn.ReLU):
                # print("taken")
                out_list.append(out) 
        # print(len(out_list))
        # exit() 
        # Flatting
        x = out
        x = x.view(x.size(0), -1)

        x = self.feat_reducer(x)

        if extra_info is not None:
            agg = torch.cat((x, extra_info), dim=1)
        else:
            agg = x

        res = self.classifier(agg)

        return res, out_list


    # # function to extract a specific features
    def intermediate_forward(self, img, layer_index):
        # See the notes in the feature_list function
        out = img
        count = 0
        for ii, layer in enumerate(*self.features[:-1]): 
            out = layer(out)
            if isinstance(layer, nn.ReLU):
                if count == layer_index: return out     
                count += 1