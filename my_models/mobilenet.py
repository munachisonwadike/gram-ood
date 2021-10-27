# -*- coding: utf-8 -*-
"""
Autor: André Pacheco
Email: pacheco.comp@gmail.com

This files implements the mobilenet + extra information

"""

import torch
from torch import nn
import torch.nn.functional as nnF
from torchvision import models



class Net (nn.Module):
    """
    This class get a resnet model, changes its FC layer and include the extra information on it (if applicable)
    """

    def __init__(self, mobilenet, num_class, freeze_conv=False, n_extra_info=0, p_dropout=0.5, neurons_class=256,
                 feat_reducer=None, classifier=None):
        """

        :param mobilenet: the resnet model from torchvision.model. Ex: model.resnet50
        :param num_class: the number of task's classes
        :param freeze_conv: if you'd like to freeze the extraction map from the model
        :param n_extra_info: the number of extra information you wanna include in the model
        :param p_dropout: the dropout probability
        """
        super(Net, self).__init__()

        self.features = nn.Sequential(*list(mobilenet.children())[:-1])

        # freezing the convolution layers
        if freeze_conv:
            for param in self.features.parameters():
                param.requires_grad = False

        # Feature reducer
        if feat_reducer is None:
            self.feat_reducer = nn.Sequential(
                nn.Linear(1280, neurons_class),
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
        x = x.mean([2, 3])

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

        count = 0
        for i, layer in enumerate(*self.features):
            # print(i, type(layer))
            out = layer(out)
            if isinstance(layer, models.mobilenet.ConvBNReLU):
                # print("taken", type(layer))
                out_list.append(out)  
            if isinstance(layer, models.mobilenet.InvertedResidual):
                if count % 2 == 0:
                    # print("taken", type(layer))
                    out_list.append(out)  
                count+= 1

        # print("OUT LIST SHAPE", len(out_list))
        # exit()
        # Flatting
        x = out
        x = x.mean([2, 3])

        x = self.feat_reducer(x)

        if extra_info is not None:
            agg = torch.cat((x, extra_info), dim=1)
        else:
            agg = x

        res = self.classifier(agg)

        return res, out_list

    # # function to extract a specific features
    def intermediate_forward(self, img, layer_index):
        # print("****DESIRED LAYER INDEX", layer_index)
        # See the notes in the feature_list function
        out = img
        index_count = 0
        count = 0
        tried_inverted = 0 #just to know if i am deeling with that convbnrelu that came before or after inverted residuals
        for ii, layer in enumerate(*self.features): 
            # print(ii, type(layer))
            out = layer(out)
            if isinstance(layer, models.mobilenet.ConvBNReLU) and (layer_index == index_count): 
                # print("****OBTAINED LAYER INDEX 1", layer_index, out.shape)
                return out 
            if isinstance(layer, models.mobilenet.InvertedResidual):
                # print("tried", count)
                tried_inverted = 1 
                if count % 2 == 0:
                    index_count += 1
                    # print("tried", layer_index, index_count)
                    if index_count == layer_index: 
                        # print("****OBTAINED LAYER INDEX 2", layer_index, out.shape)
                        index_count += 1 # just so we can move the pointer forward
                        return out    
                count += 1

            elif isinstance(layer, models.mobilenet.ConvBNReLU) and (tried_inverted == 1 ): 
                # print("****OBTAINED LAYER INDEX 3", layer_index, out.shape)
                return out
                   

