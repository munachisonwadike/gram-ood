# -*- coding: utf-8 -*-
"""
Autor: André Pacheco
Email: pacheco.comp@gmail.com

"""

import torch
from torch import nn
import torch.nn.functional as nnF
from torchvision import  models

class Net (nn.Module):

    def __init__(self, densenet, num_class, freeze_conv=False, n_extra_info=0, p_dropout=0.5, neurons_class=256,
                 feat_reducer=None, classifier=None):
        
        super(Net, self).__init__()
        
        self.features = nn.Sequential(*list(densenet.children())[:-1])

        # freezing the convolution layers
        if freeze_conv:
            for param in self.features.parameters():
                param.requires_grad = False

        if feat_reducer is None:
            self.feat_reducer = nn.Sequential(
                nn.Linear(1024, neurons_class),
                nn.BatchNorm1d(neurons_class),
                nn.ReLU(),
                nn.Dropout(p=p_dropout)
            )
        else:
            self.feat_reducer = feat_reducer

        if classifier is None:
            self.classifier = nn.Sequential(
                nn.Linear(neurons_class + n_extra_info, num_class)
            )
        else:
            self.classifier = classifier


    def forward(self, img, extra_info=None):

        xf = self.features(img)
        x = nnF.relu(xf, inplace=True)
        x = nnF.adaptive_avg_pool2d(x, (1, 1)).view(xf.size(0), -1)

        x = self.feat_reducer(x)

        if extra_info is not None:
            agg = torch.cat((x, extra_info), dim=1)
        else:
            agg = x

        x = self.classifier(agg)

        return x 

    def feature_list(self, img, extra_info=None):
        # https://discuss.pytorch.org/t/accessing-intermediate-layers-of-a-pretrained-network-forward/12113
        # using the above reference, built this function to return intermediate features
        # the other reference was the original function here: https://github.com/pokaxpoka/deep_Mahalanobis_detector/blob/90c2105e78c6f76a2801fc4c1cb1b84f4ff9af63/models/resnet.py
        # the feature_list function is essentially the forward function, returns intermediate features in `out_list` 
        # instead of full forward pass results  
        out = img
        out_list = []
        for ii, layer in enumerate(*self.features):
            # print(ii, type(layer))
            out = layer(out)
            if isinstance(layer, models.densenet._DenseBlock) or isinstance(layer, nn.ReLU) or  isinstance(layer, nn.BatchNorm2d):
                # print("taken")
                out_list.append(out) 
        
        # exit() 
        xf = out
        x = nnF.relu(xf, inplace=True)
        x = nnF.adaptive_avg_pool2d(x, (1, 1)).view(xf.size(0), -1)

        x = self.feat_reducer(x)

        if extra_info is not None:
            agg = torch.cat((x, extra_info), dim=1)
        else:
            agg = x
        x = self.classifier(agg)
        return x, out_list

    # function to extract a specific features
    def intermediate_forward(self, img, layer_index):
        # See the notes in the feature_list function
        out = img
        count = 0
        for ii, layer in enumerate(*self.features): 
            out = layer(out)
            if isinstance(layer, nn.ReLU) and layer_index == 0: return out 
            elif isinstance(layer, models.densenet._DenseBlock) or isinstance(layer, nn.BatchNorm2d):
                count += 1
                if count == layer_index: return out     
