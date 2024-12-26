import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from .conformer.conformer import Conformer
from torchvision.ops import nms
import numpy as np

convert_list = [[64, 256, 512, 1024, 1024], [32, 64, 64, 64, 64]]
config_list_mask = [[32, 0, 32, 3, 1], [64, 0, 64, 3, 1], [64, 0, 64, 5, 2], [64, 0, 64, 5, 2],
                    [64, 0, 64, 7, 3]]
config_list_edge = [[32], [64, 64, 64, 64]]

anchor_size = [8, 32, 128]
aspect_ratio = [0.5, 1, 1.5, 2, 2.5]

convert_detect = [32, 64, 128, 256, 512]


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


def generate_anchor_boxes(anchor_sizes, aspect_ratios, dtype=torch.float32):
    """
    生成基于不同尺寸和宽高比的锚点。

    参数:
    - anchor_sizes: 锚点的尺寸列表。
    - aspect_ratios: 宽高比列表。
    - dtype: 数据类型。

    返回:
    - 一个形状为 [len(anchor_sizes) * len(aspect_ratios), 4] 的张量，
      表示锚点的坐标，以中心为原点，格式为 [x_min, y_min, x_max, y_max]。
    """
    anchors = []
    for size in anchor_sizes:
        area = size ** 2
        for ratio in aspect_ratios:
            # 宽度和高度
            width = np.sqrt(area / ratio)
            height = np.sqrt(area * ratio)

            # 生成以原点为中心的锚点 [x_min, y_min, x_max, y_max]
            x_min = -width / 2
            y_min = -height / 2
            x_max = width / 2
            y_max = height / 2

            anchors.append([x_min, y_min, x_max, y_max])

    return torch.tensor(anchors, dtype=dtype).cuda()


def compute_stride():
    return 32


def apply_bbox_deltas(anchors, deltas):
    """
    应用边界框回归偏移量来调整锚点框。

    参数:
    - anchors: [N, 4]的张量，表示N个锚点框的[x_min, y_min, x_max, y_max]。
    - deltas: [N, 4]的张量，表示N个偏移量[dx, dy, dw, dh]。

    返回:
    - 调整后的边界框，同样是[N, 4]的格式。
    """
    widths = anchors[:, 2] - anchors[:, 0] + 1.0
    heights = anchors[:, 3] - anchors[:, 1] + 1.0
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    # 中心坐标的偏移
    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    # 尺寸的相对变化
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    # 转换回[x_min, y_min, x_max, y_max]格式
    pred_boxes = torch.zeros_like(deltas)
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w - 1.0
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h - 1.0

    return pred_boxes/541


class RPN(nn.Module):
    """区域提议网络 (RPN)"""

    def __init__(self, in_channels, anchor_sizes, aspect_ratios):
        super(RPN, self).__init__()
        self.anchor_sizes = anchor_sizes
        self.aspect_ratios = aspect_ratios

        # 生成anchor的数量
        self.num_anchors = len(anchor_sizes) * len(aspect_ratios)

        # 卷积层用于处理输入的特征图
        self.conv = nn.Conv2d(in_channels, 512, 3, 1, 1)

        # 分类层：预测anchor属于前景或背景
        self.cls_logits = nn.Conv2d(512, self.num_anchors * 2, 1)

        # 回归层：预测anchor到真实边框的偏移量
        self.bbox_pred = nn.Conv2d(512, self.num_anchors * 4, 1)

    def forward(self, features):
        # 特征图处理
        features = F.relu(self.conv(features))
        cls_logits = self.cls_logits(features)
        bbox_preds = self.bbox_pred(features)

        # 应用sigmoid函数于cls_logits以获得每个锚点的置信度得分
        cls_scores = torch.sigmoid(cls_logits)

        # 动态生成锚点
        dtype, device = features.dtype, features.device
        stride = compute_stride()
        anchors = self.generate_anchors(features, stride, dtype)

        return cls_scores, bbox_preds, anchors

    def generate_anchors(self, features, stride, dtype):
        # 特征图尺寸
        height, width = features.shape[2], features.shape[3]
        # 生成基础锚点
        base_anchors = generate_anchor_boxes(anchor_sizes=self.anchor_sizes, aspect_ratios=self.aspect_ratios,
                                             dtype=dtype)
        # 计算网格中心点
        grid_height, grid_width = height * stride, width * stride
        shifts_x = torch.arange(0, grid_width, step=stride, dtype=dtype).cuda()
        shifts_y = torch.arange(0, grid_height, step=stride, dtype=dtype).cuda()
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

        # 在每个网格位置重复基础锚点，并应用偏移
        all_anchors = (base_anchors.view(-1, 4) + shifts.view(-1, 1, 4)).reshape(-1, 4)
        return all_anchors


def apply_nms(cls_logits, bbox_preds, anchors, score_threshold=0.5, iou_threshold=0.1, max_detections=30):
    batch_size, _, H, W = cls_logits.shape
    A = bbox_preds.size(1) // 4  # 每个位置的锚点数量

    all_selected_boxes = []
    all_selected_scores = []
    all_selected_deltas = []

    # 处理每个批次
    for batch in range(batch_size):
        # 应用偏移量到锚点上，得到调整后的预测边界框
        delta_preds = bbox_preds[batch].reshape(-1, 4)
        adjusted_boxes = apply_bbox_deltas(anchors, delta_preds)
        adjusted_boxes = torch.clamp(adjusted_boxes, min=0.0, max=1.0)

        # 提取前景分数，并重塑为一维数组
        cls_scores = torch.sigmoid(cls_logits[batch]).reshape(-1, 2)[:, 1]  # 假设前景类别位于索引1

        # 筛选得分高于阈值的预测
        high_score_idxs = cls_scores > score_threshold
        scores_high = cls_scores[high_score_idxs]
        boxes_high = adjusted_boxes[high_score_idxs]
        delta_high = delta_preds[high_score_idxs]

        # 应用NMS
        nms_idxs = nms(boxes_high, scores_high, iou_threshold)

        if len(nms_idxs) > max_detections:
            top_scores, top_scores_idx = scores_high[nms_idxs].topk(max_detections)
            nms_idxs = nms_idxs[top_scores_idx]

        # 保存选中的边界框和得分
        selected_boxes = boxes_high[nms_idxs]
        selected_scores = scores_high[nms_idxs]
        select_deltas = delta_high[nms_idxs]

        all_selected_boxes.append(selected_boxes)
        all_selected_scores.append(selected_scores)
        all_selected_deltas.append(select_deltas)

    return all_selected_boxes, all_selected_scores, all_selected_deltas


# dual branch seg
class TCSeg(nn.Module):
    def __init__(self, config_list_mask, config_list_edge):
        super(TCSeg, self).__init__()
        self.relu = nn.ReLU()
        self.config_list_mask = config_list_mask
        self.config_list_edge = config_list_edge

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
            s_c_merge.append(nn.Sequential(nn.Conv2d(64 * (i + 2), 64, 1, 1, 1),
                                           nn.BatchNorm2d(64), nn.ReLU(inplace=True)))

        self.s_c_merge = nn.ModuleList(s_c_merge)

        self.c_edge_conv = nn.ModuleList([nn.Sequential(nn.Conv2d(64, 32, 1, 1, bias=False), nn.ReLU(inplace=True))])

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

        # transformer branch
        self.trans_convert = nn.Sequential(nn.Conv2d(768, 64, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.trans_up = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
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
        t_ch = 32
        self.t_up = nn.Sequential(nn.Conv2d(t_ch, t_ch, 7, 1, 3), nn.BatchNorm2d(t_ch),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(t_ch, t_ch, 7, 1, 3), nn.BatchNorm2d(t_ch),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(t_ch, t_ch, 7, 1, 3), nn.BatchNorm2d(t_ch),
                                  nn.ReLU(inplace=True))

        self.cls1 = nn.Sequential(nn.Conv2d(32, 128, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(128),
                                  nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1))
        self.cls2 = nn.Sequential(nn.Linear(128, 64),
                                  nn.Linear(64, 3))

        detect_con = []

        for i in range(len(convert_detect)-1):
            detect_con.append(
                nn.Sequential(nn.Conv2d(convert_detect[i], convert_detect[i+1], kernel_size=3, stride=2, padding=1, bias=False),
                              nn.BatchNorm2d(convert_detect[i+1]),
                              nn.ReLU(inplace=True)))

        self.detect_con = nn.ModuleList(detect_con)

        self.trans_channel = 512
        self.rpn = RPN(self.trans_channel, anchor_sizes=anchor_size, aspect_ratios=aspect_ratio)

    def forward(self, c_feature, t_feature, x_size):
        f_mask, f_edge, trans_feature = [], [], []
        edge_out, mask_out, cls_out = [], [], []

        # edge block
        feature_0 = self.c_mask_conv[-1](c_feature[-1])
        edge_d_feature = self.c_mask_conv[0](
            c_feature[0] + F.interpolate((self.c_edge_conv[-1](feature_0)), c_feature[0].size()[2:], mode='bilinear',
                                         align_corners=True))
        f_edge.append(edge_d_feature)
        edge_out.append(F.interpolate(self.up_edge(edge_d_feature), x_size, mode='bilinear', align_corners=True))

        # transformer block
        B, _, C = t_feature.shape
        x_t = t_feature[:, 1:].transpose(1, 2).reshape(B, C, 34, 34)
        # 896 56 56  541 34 34
        x_trans = self.trans_convert(x_t)
        for i in range(3):
            trans_feature.append(x_trans)
            x_trans = self.trans_up(x_trans)

        for i, i_x in enumerate(f_edge):
            tmp = F.interpolate(self.c_edge_conv[-1](x_trans), i_x.size()[2:], mode='bilinear',
                                align_corners=True) + i_x
            tmp = self.t_up(tmp)
            tmp = F.interpolate(tmp, x_trans.size()[2:], mode='bilinear', align_corners=True)
            tmp_tr = tmp
            tmp = self.cls1(tmp).flatten(1)
            tmp = self.cls2(tmp)
            cls_out.append(tmp)
        mask_out.append(F.interpolate(self.up_mask_final(tmp_tr), x_size, mode='bilinear', align_corners=True))

        detect_convert = self.detect_con[0](tmp_tr)
        for i in range(1, 4):
            detect_convert = self.detect_con[i](detect_convert)

        # detection layer
        cls_scores, bbox_preds, anchors = self.rpn(detect_convert)

        final_boxes, final_scores, final_delta = apply_nms(cls_scores, bbox_preds, anchors)

        # cnn branch
        f_tmp = [feature_0]
        f_mask.append(feature_0)
        mask_out.append(F.interpolate(self.up_mask(feature_0), x_size, mode='bilinear', align_corners=True))

        f_tmp.append(self.s_c_merge[0](torch.cat(
            [trans_feature[0], F.interpolate(f_tmp[0], c_feature[3].size()[2:], mode='bilinear', align_corners=True),
             c_feature[3]], dim=1)))
        f_mask.append(self.c_mask_conv[3](f_tmp[-1]))
        mask_out.append(F.interpolate(self.up_mask(f_mask[-1]), x_size, mode='bilinear', align_corners=True))

        f_up1 = F.interpolate(f_tmp[1], c_feature[2].size()[2:], mode='bilinear', align_corners=True)
        f_up2 = F.interpolate(f_tmp[0], c_feature[2].size()[2:], mode='bilinear', align_corners=True)
        f_tmp.append(self.s_c_merge[1](torch.cat([trans_feature[1], f_up1, f_up2, c_feature[2]], dim=1)))
        f_mask.append(self.c_mask_conv[2](f_tmp[-1]))
        mask_out.append(F.interpolate(self.up_mask(f_mask[-1]), x_size, mode='bilinear', align_corners=True))

        f_up3 = F.interpolate(f_tmp[2], c_feature[1].size()[2:], mode='bilinear', align_corners=True)
        f_up4 = F.interpolate(f_tmp[1], c_feature[1].size()[2:], mode='bilinear', align_corners=True)
        f_up5 = F.interpolate(f_tmp[0], c_feature[1].size()[2:], mode='bilinear', align_corners=True)
        f_tmp.append(self.s_c_merge[2](torch.cat([trans_feature[2], f_up3, f_up4, f_up5, c_feature[1]], dim=1)))
        f_mask.append(self.c_mask_conv[1](f_tmp[-1]))
        mask_out.append(F.interpolate(self.up_mask(f_mask[-1]), x_size, mode='bilinear', align_corners=True))

        # merge with edge
        tmp_feature = []
        for i, i_x in enumerate(f_edge):
            for j, j_x in enumerate(f_mask):
                tmp = F.interpolate(self.c_edge_conv[-1](j_x), i_x.size()[2:], mode='bilinear',
                                    align_corners=True) + i_x
                tmp_f = self.m_e_merge_conv[0][j](tmp)
                mask_out.append(F.interpolate(self.up_mask_final(tmp_f), x_size, mode='bilinear', align_corners=True))
                tmp_feature.append(tmp_f)

        for i_x in tmp_feature:
            i_x = F.interpolate(i_x, x_trans.size()[2:], mode='bilinear', align_corners=True)
            tmp = self.cls1(i_x).flatten(1)
            tmp = self.cls2(tmp)
            cls_out.append(tmp)

        tmp_fea = tmp_feature[0]

        for i_fea in range(len(tmp_feature) - 1):
            tmp_fea = self.relu(torch.add(tmp_fea, F.interpolate((tmp_feature[i_fea + 1]), tmp_feature[0].size()[2:],
                                                                 mode='bilinear', align_corners=True)))

        mask_out.append(F.interpolate(self.up_mask_final(tmp_fea), x_size, mode='bilinear', align_corners=True))

        return mask_out, cls_out, final_boxes, final_scores, final_delta


# TUN network
class TUN_bone(nn.Module):
    def __init__(self, model, base):
        super(TUN_bone, self).__init__()
        self.model = model(config_list_mask, config_list_edge)
        self.base = base
        self.convert = ConvertLayer(convert_list)

    def forward(self, x):
        x_size = x.size()[2:]
        conv2merge, x_t = self.base(x)
        conv2merge = self.convert(conv2merge)
        mask_out, cls_out, final_boxes, final_scores, final_delta = self.model(conv2merge, x_t, x_size)
        return mask_out, cls_out, final_boxes, final_scores, final_delta


# build the whole network
def build_model(ema=False):
    if not ema:
        return TUN_bone(TCSeg, Conformer())
    else:
        return TUN_bone(TCSeg, Conformer())


# weight init
def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()
