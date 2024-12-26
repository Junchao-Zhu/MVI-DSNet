import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from .conformer.conformer import Conformer


convert_list = [[64, 256, 512, 1024, 1024], [32, 64, 64, 64, 64]]
config_list_mask = [[32, 0, 32, 3, 1], [64, 0, 64, 3, 1], [64, 0, 64, 5, 2], [64, 0, 64, 5, 2],
                    [64, 0, 64, 7, 3]]
config_list_edge = [[32], [64, 64, 64, 64]]


class ConvertLayer(nn.Module):
    def __init__(self, list_convert):
        super(ConvertLayer, self).__init__()

        conv_0 = []
        for i in range(len(list_convert[0])):
            conv_0.append(
                nn.Sequential(nn.Conv2d(list_convert[0][i], list_convert[1][i], 1, 1, bias=False),
                              nn.BatchNorm2d(list_convert[1][i]),
                              nn.ReLU(inplace=True)))

        self.convert0 = nn.ModuleList(conv_0)

    def forward(self, list_x):
        resl = []
        for i in range(len(list_x)):
            resl.append(self.convert0[i](list_x[i]))
        return resl


# dual branch seg
class TCSeg(nn.Module):
    def __init__(self, base=Conformer()):
        super(TCSeg, self).__init__()
        self.relu = nn.ReLU()
        self.config_list_mask = config_list_mask
        self.config_list_edge = config_list_edge
        self.base = base
        self.convert = ConvertLayer(convert_list)

        # cnn branch decoder for each layer
        c_mask_conv = []
        for i, config in enumerate(self.config_list_mask):
            c_mask_conv.append(
                nn.Sequential(nn.Conv2d(config[0], config[2], config[3], 1, config[4]), nn.BatchNorm2d(config[2]),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(config[2], config[2], config[3], 1, config[4]), nn.BatchNorm2d(config[2]),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(config[2], config[2], config[3], 1, config[4]), nn.BatchNorm2d(config[2]),
                              nn.ReLU(inplace=True)))

        self.c_mask_conv = nn.ModuleList(c_mask_conv)

        s_c_merge = []
        for i in range(1, len(self.config_list_mask) - 1):
            s_c_merge.append(nn.Sequential(nn.Conv2d(64 * (i + 1), 64, 1, 1, 1),
                                           nn.BatchNorm2d(64), nn.ReLU(inplace=True)))

        self.s_c_merge = nn.ModuleList(s_c_merge)

        self.c_edge_conv = nn.ModuleList([nn.Sequential(nn.Conv2d(128, 32, 1, 1, bias=False), nn.ReLU(inplace=True)), nn.Sequential(nn.Conv2d(64, 32, 1, 1, bias=False), nn.ReLU(inplace=True))])

        self.up_mask = nn.Sequential(
            nn.Conv2d(64, 16, 3, 1, 1), nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Conv2d(16, 1, 1)
        )

        self.up_edge = nn.Sequential(
            nn.Conv2d(32, 8, 3, 1, 1), nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Conv2d(8, 1, 1)
        )

        # mask & edge merge
        feature_config_list = [[3, 1], [5, 2], [5, 2], [7, 3]]
        self.mask_convert = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(64, 32, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))])
        self.m_e_merge_conv = nn.ModuleList([
            nn.Sequential(nn.Conv2d(32, 32, feature_config_list[idx][0], 1, feature_config_list[idx][1]),
                          nn.BatchNorm2d(32),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(32, 32, feature_config_list[idx][0], 1, feature_config_list[idx][1]),
                          nn.BatchNorm2d(32),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(32, 32, feature_config_list[idx][0], 1, feature_config_list[idx][1]),
                          nn.BatchNorm2d(32),
                          nn.ReLU(inplace=True)) for idx in range(len(feature_config_list))
        ])

        self.up_mask_final = nn.Sequential(
            nn.Conv2d(32, 8, 3, 1, 1), nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Conv2d(8, 1, 1))

        # seg to cls
        self.cls1 = nn.Sequential(nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(8),
                                  nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1))
        self.cls2 = nn.Sequential(nn.Linear(8, 3))

    def forward(self, x):
        x_size = x.size()[2:]
        conv2merge, x_t = self.base(x)
        conv2merge = self.convert(conv2merge)

        c_feature = conv2merge
        t_feature = x_t
        x_size = x_size

        f_mask, f_edge, trans_feature, cnn_feature = [], [], [], []
        mask_out, cls_out = [], []

        # edge block
        feature_0 = self.c_mask_conv[-1](c_feature[-1])
        edge_d_feature = self.c_mask_conv[0](
            c_feature[0] + F.interpolate((self.c_edge_conv[-1](feature_0)), c_feature[0].size()[2:], mode='bilinear',
                                         align_corners=True))
        f_edge.append(edge_d_feature)

        # cnn branch
        f_tmp = [feature_0]
        f_mask.append(feature_0)

        f_tmp.append(self.s_c_merge[0](torch.cat(
            [F.interpolate(f_tmp[0], c_feature[3].size()[2:], mode='bilinear', align_corners=True),
             c_feature[3]], dim=1)))

        cnn_feature.append(torch.cat(
            [F.interpolate(f_tmp[0], c_feature[3].size()[2:], mode='bilinear', align_corners=True),
             c_feature[3]], dim=1))

        f_mask.append(self.c_mask_conv[3](f_tmp[-1]))

        f_up1 = F.interpolate(f_tmp[1], c_feature[2].size()[2:], mode='bilinear', align_corners=True)
        f_up2 = F.interpolate(f_tmp[0], c_feature[2].size()[2:], mode='bilinear', align_corners=True)
        f_tmp.append(self.s_c_merge[1](torch.cat([f_up1, f_up2, c_feature[2]], dim=1)))
        cnn_feature.append(torch.cat([f_up1, f_up2, c_feature[2]], dim=1))

        f_mask.append(self.c_mask_conv[2](f_tmp[-1]))

        f_up3 = F.interpolate(f_tmp[2], c_feature[1].size()[2:], mode='bilinear', align_corners=True)
        f_up4 = F.interpolate(f_tmp[1], c_feature[1].size()[2:], mode='bilinear', align_corners=True)
        f_up5 = F.interpolate(f_tmp[0], c_feature[1].size()[2:], mode='bilinear', align_corners=True)
        f_tmp.append(self.s_c_merge[2](torch.cat([f_up3, f_up4, f_up5, c_feature[1]], dim=1)))
        cnn_feature.append(torch.cat([f_up3, f_up4, f_up5, c_feature[1]], dim=1))

        f_mask.append(self.c_mask_conv[1](f_tmp[-1]))

        # merge with edge to get better mask result
        tmp_feature = []
        for i, i_x in enumerate(f_edge):
            for j, j_x in enumerate(f_mask):
                tmp = F.interpolate(self.c_edge_conv[-1](j_x), i_x.size()[2:], mode='bilinear',
                                    align_corners=True) + i_x
                tmp_f = self.m_e_merge_conv[0][j](tmp)
                mask_out.append(F.interpolate(self.up_mask_final(tmp_f), x_size, mode='bilinear', align_corners=True))
                tmp_feature.append(tmp_f)

        tmp_fea = tmp_feature[0]
        for i_fea in range(len(tmp_feature) - 1):
            tmp_fea = self.relu(torch.add(tmp_fea, F.interpolate((tmp_feature[i_fea + 1]), tmp_feature[0].size()[2:],
                                                                 mode='bilinear', align_corners=True)))

        mask_out.append(F.interpolate(self.up_mask_final(tmp_fea), x_size, mode='bilinear', align_corners=True))

        # transformer block
        B, _, C = t_feature.shape

        x_t = t_feature[:, 1:].transpose(1, 2).reshape(B, int(C / 16), 160, 160)

        x_trans = torch.cat([x_t, c_feature[1]], dim=1)

        for i_x in tmp_feature:
            tmp = self.cls1(i_x).flatten(1)
            tmp = self.cls2(tmp)
            cls_out.append(tmp)

        return x_trans, c_feature[2], mask_out, cls_out


# weight init
def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()