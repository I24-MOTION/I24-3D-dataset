import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

_weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict

class KitModel(nn.Module):

    
    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global _weights_dict
        _weights_dict = load_weights(weight_file)

        self.conv2d_31 = self.__conv(2, name='conv2d_31', in_channels=3, out_channels=16, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_29 = self.__batch_normalization(2, 'batch_normalization_29', num_features=16, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_32 = self.__conv(2, name='conv2d_32', in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_30 = self.__batch_normalization(2, 'batch_normalization_30', num_features=16, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_33 = self.__conv(2, name='conv2d_33', in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_31 = self.__batch_normalization(2, 'batch_normalization_31', num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_34 = self.__conv(2, name='conv2d_34', in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_32 = self.__batch_normalization(2, 'batch_normalization_32', num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_35 = self.__conv(2, name='conv2d_35', in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_33 = self.__batch_normalization(2, 'batch_normalization_33', num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_36 = self.__conv(2, name='conv2d_36', in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_34 = self.__batch_normalization(2, 'batch_normalization_34', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_37 = self.__conv(2, name='conv2d_37', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_35 = self.__batch_normalization(2, 'batch_normalization_35', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_38 = self.__conv(2, name='conv2d_38', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_36 = self.__batch_normalization(2, 'batch_normalization_36', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_39 = self.__conv(2, name='conv2d_39', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_37 = self.__batch_normalization(2, 'batch_normalization_37', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_40 = self.__conv(2, name='conv2d_40', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_38 = self.__batch_normalization(2, 'batch_normalization_38', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_41 = self.__conv(2, name='conv2d_41', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_39 = self.__batch_normalization(2, 'batch_normalization_39', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_42 = self.__conv(2, name='conv2d_42', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_40 = self.__batch_normalization(2, 'batch_normalization_40', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_43 = self.__conv(2, name='conv2d_43', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_41 = self.__batch_normalization(2, 'batch_normalization_41', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_44 = self.__conv(2, name='conv2d_44', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_42 = self.__batch_normalization(2, 'batch_normalization_42', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_45 = self.__conv(2, name='conv2d_45', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_43 = self.__batch_normalization(2, 'batch_normalization_43', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_46 = self.__conv(2, name='conv2d_46', in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_44 = self.__batch_normalization(2, 'batch_normalization_44', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_47 = self.__conv(2, name='conv2d_47', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_45 = self.__batch_normalization(2, 'batch_normalization_45', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_48 = self.__conv(2, name='conv2d_48', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_46 = self.__batch_normalization(2, 'batch_normalization_46', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_49 = self.__conv(2, name='conv2d_49', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_47 = self.__batch_normalization(2, 'batch_normalization_47', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_50 = self.__conv(2, name='conv2d_50', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_48 = self.__batch_normalization(2, 'batch_normalization_48', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_51 = self.__conv(2, name='conv2d_51', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_49 = self.__batch_normalization(2, 'batch_normalization_49', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_52 = self.__conv(2, name='conv2d_52', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_50 = self.__batch_normalization(2, 'batch_normalization_50', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_53 = self.__conv(2, name='conv2d_53', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_51 = self.__batch_normalization(2, 'batch_normalization_51', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_54 = self.__conv(2, name='conv2d_54', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_52 = self.__batch_normalization(2, 'batch_normalization_52', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_55 = self.__conv(2, name='conv2d_55', in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv2d_58 = self.__conv(2, name='conv2d_58', in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_53 = self.__batch_normalization(2, 'batch_normalization_53', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.batch_normalization_55 = self.__batch_normalization(2, 'batch_normalization_55', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_56 = self.__conv(2, name='conv2d_56', in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv2d_59 = self.__conv(2, name='conv2d_59', in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_54 = self.__batch_normalization(2, 'batch_normalization_54', num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.batch_normalization_56 = self.__batch_normalization(2, 'batch_normalization_56', num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_57 = self.__conv(2, name='conv2d_57', in_channels=32, out_channels=1, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv2d_60 = self.__conv(2, name='conv2d_60', in_channels=32, out_channels=6, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)

    def forward(self, x):
        conv2d_31_pad   = F.pad(x, (1, 1, 1, 1))
        conv2d_31       = self.conv2d_31(conv2d_31_pad)
        batch_normalization_29 = self.batch_normalization_29(conv2d_31)
        activation_29   = F.relu(batch_normalization_29)
        conv2d_32_pad   = F.pad(activation_29, (1, 1, 1, 1))
        conv2d_32       = self.conv2d_32(conv2d_32_pad)
        batch_normalization_30 = self.batch_normalization_30(conv2d_32)
        activation_30   = F.relu(batch_normalization_30)
        max_pooling2d_5, max_pooling2d_5_idx = F.max_pool2d(activation_30, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        conv2d_33_pad   = F.pad(max_pooling2d_5, (1, 1, 1, 1))
        conv2d_33       = self.conv2d_33(conv2d_33_pad)
        batch_normalization_31 = self.batch_normalization_31(conv2d_33)
        activation_31   = F.relu(batch_normalization_31)
        conv2d_34_pad   = F.pad(activation_31, (1, 1, 1, 1))
        conv2d_34       = self.conv2d_34(conv2d_34_pad)
        batch_normalization_32 = self.batch_normalization_32(conv2d_34)
        activation_32   = F.relu(batch_normalization_32)
        conv2d_35_pad   = F.pad(activation_32, (1, 1, 1, 1))
        conv2d_35       = self.conv2d_35(conv2d_35_pad)
        batch_normalization_33 = self.batch_normalization_33(conv2d_35)
        add_10          = batch_normalization_33 + activation_31
        activation_33   = F.relu(add_10)
        max_pooling2d_6, max_pooling2d_6_idx = F.max_pool2d(activation_33, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        conv2d_36_pad   = F.pad(max_pooling2d_6, (1, 1, 1, 1))
        conv2d_36       = self.conv2d_36(conv2d_36_pad)
        batch_normalization_34 = self.batch_normalization_34(conv2d_36)
        activation_34   = F.relu(batch_normalization_34)
        conv2d_37_pad   = F.pad(activation_34, (1, 1, 1, 1))
        conv2d_37       = self.conv2d_37(conv2d_37_pad)
        batch_normalization_35 = self.batch_normalization_35(conv2d_37)
        activation_35   = F.relu(batch_normalization_35)
        conv2d_38_pad   = F.pad(activation_35, (1, 1, 1, 1))
        conv2d_38       = self.conv2d_38(conv2d_38_pad)
        batch_normalization_36 = self.batch_normalization_36(conv2d_38)
        add_11          = batch_normalization_36 + activation_34
        activation_36   = F.relu(add_11)
        conv2d_39_pad   = F.pad(activation_36, (1, 1, 1, 1))
        conv2d_39       = self.conv2d_39(conv2d_39_pad)
        batch_normalization_37 = self.batch_normalization_37(conv2d_39)
        activation_37   = F.relu(batch_normalization_37)
        conv2d_40_pad   = F.pad(activation_37, (1, 1, 1, 1))
        conv2d_40       = self.conv2d_40(conv2d_40_pad)
        batch_normalization_38 = self.batch_normalization_38(conv2d_40)
        add_12          = batch_normalization_38 + activation_36
        activation_38   = F.relu(add_12)
        max_pooling2d_7, max_pooling2d_7_idx = F.max_pool2d(activation_38, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        conv2d_41_pad   = F.pad(max_pooling2d_7, (1, 1, 1, 1))
        conv2d_41       = self.conv2d_41(conv2d_41_pad)
        batch_normalization_39 = self.batch_normalization_39(conv2d_41)
        activation_39   = F.relu(batch_normalization_39)
        conv2d_42_pad   = F.pad(activation_39, (1, 1, 1, 1))
        conv2d_42       = self.conv2d_42(conv2d_42_pad)
        batch_normalization_40 = self.batch_normalization_40(conv2d_42)
        activation_40   = F.relu(batch_normalization_40)
        conv2d_43_pad   = F.pad(activation_40, (1, 1, 1, 1))
        conv2d_43       = self.conv2d_43(conv2d_43_pad)
        batch_normalization_41 = self.batch_normalization_41(conv2d_43)
        add_13          = batch_normalization_41 + activation_39
        activation_41   = F.relu(add_13)
        conv2d_44_pad   = F.pad(activation_41, (1, 1, 1, 1))
        conv2d_44       = self.conv2d_44(conv2d_44_pad)
        batch_normalization_42 = self.batch_normalization_42(conv2d_44)
        activation_42   = F.relu(batch_normalization_42)
        conv2d_45_pad   = F.pad(activation_42, (1, 1, 1, 1))
        conv2d_45       = self.conv2d_45(conv2d_45_pad)
        batch_normalization_43 = self.batch_normalization_43(conv2d_45)
        add_14          = batch_normalization_43 + activation_41
        activation_43   = F.relu(add_14)
        max_pooling2d_8, max_pooling2d_8_idx = F.max_pool2d(activation_43, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        conv2d_46_pad   = F.pad(max_pooling2d_8, (1, 1, 1, 1))
        conv2d_46       = self.conv2d_46(conv2d_46_pad)
        batch_normalization_44 = self.batch_normalization_44(conv2d_46)
        activation_44   = F.relu(batch_normalization_44)
        conv2d_47_pad   = F.pad(activation_44, (1, 1, 1, 1))
        conv2d_47       = self.conv2d_47(conv2d_47_pad)
        batch_normalization_45 = self.batch_normalization_45(conv2d_47)
        activation_45   = F.relu(batch_normalization_45)
        conv2d_48_pad   = F.pad(activation_45, (1, 1, 1, 1))
        conv2d_48       = self.conv2d_48(conv2d_48_pad)
        batch_normalization_46 = self.batch_normalization_46(conv2d_48)
        add_15          = batch_normalization_46 + activation_44
        activation_46   = F.relu(add_15)
        conv2d_49_pad   = F.pad(activation_46, (1, 1, 1, 1))
        conv2d_49       = self.conv2d_49(conv2d_49_pad)
        batch_normalization_47 = self.batch_normalization_47(conv2d_49)
        activation_47   = F.relu(batch_normalization_47)
        conv2d_50_pad   = F.pad(activation_47, (1, 1, 1, 1))
        conv2d_50       = self.conv2d_50(conv2d_50_pad)
        batch_normalization_48 = self.batch_normalization_48(conv2d_50)
        add_16          = batch_normalization_48 + activation_46
        activation_48   = F.relu(add_16)
        conv2d_51_pad   = F.pad(activation_48, (1, 1, 1, 1))
        conv2d_51       = self.conv2d_51(conv2d_51_pad)
        batch_normalization_49 = self.batch_normalization_49(conv2d_51)
        activation_49   = F.relu(batch_normalization_49)
        conv2d_52_pad   = F.pad(activation_49, (1, 1, 1, 1))
        conv2d_52       = self.conv2d_52(conv2d_52_pad)
        batch_normalization_50 = self.batch_normalization_50(conv2d_52)
        add_17          = batch_normalization_50 + activation_48
        activation_50   = F.relu(add_17)
        conv2d_53_pad   = F.pad(activation_50, (1, 1, 1, 1))
        conv2d_53       = self.conv2d_53(conv2d_53_pad)
        batch_normalization_51 = self.batch_normalization_51(conv2d_53)
        activation_51   = F.relu(batch_normalization_51)
        conv2d_54_pad   = F.pad(activation_51, (1, 1, 1, 1))
        conv2d_54       = self.conv2d_54(conv2d_54_pad)
        batch_normalization_52 = self.batch_normalization_52(conv2d_54)
        add_18          = batch_normalization_52 + activation_50
        activation_52   = F.relu(add_18)
        conv2d_55_pad   = F.pad(activation_52, (1, 1, 1, 1))
        conv2d_55       = self.conv2d_55(conv2d_55_pad)
        conv2d_58_pad   = F.pad(activation_52, (1, 1, 1, 1))
        conv2d_58       = self.conv2d_58(conv2d_58_pad)
        batch_normalization_53 = self.batch_normalization_53(conv2d_55)
        batch_normalization_55 = self.batch_normalization_55(conv2d_58)
        activation_53   = F.relu(batch_normalization_53)
        activation_55   = F.relu(batch_normalization_55)
        conv2d_56_pad   = F.pad(activation_53, (1, 1, 1, 1))
        conv2d_56       = self.conv2d_56(conv2d_56_pad)
        conv2d_59_pad   = F.pad(activation_55, (1, 1, 1, 1))
        conv2d_59       = self.conv2d_59(conv2d_59_pad)
        activation_54 = self.batch_normalization_54(conv2d_56)
        activation_56 = self.batch_normalization_56(conv2d_59)
        conv2d_57_pad   = F.pad(activation_54, (1, 1, 1, 1))
        conv2d_57       = self.conv2d_57(conv2d_57_pad)
        conv2d_60_pad   = F.pad(activation_56, (1, 1, 1, 1))
        conv2d_60       = self.conv2d_60(conv2d_60_pad)
        conv2d_57_activation = F.sigmoid(conv2d_57)
        concatenate_2   = torch.cat((conv2d_57_activation, conv2d_60,), 1)
        return concatenate_2


    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['weights']))
        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if   dim == 0 or dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()

        if 'scale' in _weights_dict[name]:
            layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['scale']))
        else:
            layer.weight.data.fill_(1)

        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        else:
            layer.bias.data.fill_(0)

        layer.state_dict()['running_mean'].copy_(torch.from_numpy(_weights_dict[name]['mean']))
        layer.state_dict()['running_var'].copy_(torch.from_numpy(_weights_dict[name]['var']))
        return layer

