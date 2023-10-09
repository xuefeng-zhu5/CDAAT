import torch
from torch import nn
import os
from lib.utils.box_ops import box_xyxy_to_cxcywh, box_xyxy_to_xywh
from lib.utils.misc import NestedTensor
from .backbone import build_backbone
from .head import build_box_head
from .transformer import build_transformer
import cv2
import numpy as np
from .cdaam import CDAAM

from lib.utils.merge import merge_feature_sequence
from lib.utils.misc import NestedTensor
import matplotlib.pyplot as plt

def fast_color_histogram(image, color_stride, mask):
    # input:image is a 3d array, shape[W][H][C]
    hist = torch.arange((256 // color_stride) ** 3).reshape(256 // color_stride, 256 // color_stride,
                                                            256 // color_stride)
    H, W, C = image.shape
    image_index = torch.tensor(image, dtype=torch.long) // color_stride
    index = image_index.reshape(-1, C).T
    hist = torch.bincount(hist[index[0], index[1], index[2]], weights=mask.reshape(-1, ).T, minlength=(256 // color_stride) ** 3)
    hist_3 = hist.reshape(256 // color_stride, 256 // color_stride, 256 // color_stride) / (np.sum(np.uint8(mask)) + 1)
    return hist_3


class BASETRACK(nn.Module):
    """
    This is the base class for Transformer Tracking.
    """

    def __init__(self, backbone, transformer, box_head, num_queries,
                 aux_loss=False, head_type='CORNER'):
        """
        Initializes the model.

        Args:
            backbone: Torch module of the backbone to be used. See backbone.py
            transformer: Torch module of the transformer architecture. See transformer.py
            num_queries: Number of object queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """

        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.box_head = box_head
        self.num_queries = num_queries
        self.hidden_dim = transformer.d_model
        self.foreground_embed = nn.Embedding(num_queries, self.hidden_dim)
        self.background_embed = nn.Embedding(num_queries, self.hidden_dim)
        self.bottleneck = nn.Conv2d(backbone.num_channels, self.hidden_dim, kernel_size=(1, 1))  # The bottleneck layer
        self.cdaam_color = CDAAM(inchannel=1, outchannel=self.hidden_dim)
        self.cdaam_depth = CDAAM(inchannel=1, outchannel=self.hidden_dim)

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == 'CORNER':
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)
        self.pool_sz = 4
        self.pool_len = self.pool_sz ** 2

    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord):
        """
        This is a workaround to make torchscript happy, as torchscript
        doesn't support dictionary with non-homogeneous values, such
        as a dict having both a Tensor and a list.
        """

        return [{'pred_boxes': b}
                for b in outputs_coord[:-1]]


class CDAATRACK(BASETRACK):
    """
    This is the class for Dual Graph Tracking.
    """

    def __init__(self, backbone, transformer, box_head, num_queries,
                 aux_loss=False, head_type='CORNER', iou_head=None):
        """
        Initializes the model.

        Args:
            backbone: Torch module of the backbone to be used. See backbone.py
            transformer: Torch module of the transformer architecture. See transformer.py
            num_queries: Number of object queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """

        super().__init__(backbone, transformer, box_head, num_queries, aux_loss=aux_loss, head_type=head_type)


    def forward(self, data=None, search_dic=None, refer_dic=None, refer_reg=None, out_embed=None,
                mode='backbone'):
        if mode == 'backbone':
            return self.forward_backbone(data)
        elif mode == 'transformer':
            return self.forward_transformer(search_dic, refer_dic, refer_reg)
        elif mode == 'head':
            return self.forward_box_head(out_embed)
        else:
            raise ValueError

    def forward_backbone(self, data):

        ############################################## get ResNet features #############################################
        search_dict_list = []
        search_img = data['search_images'].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        search_att = data['search_att'].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)

        # Forward the backbone # search region
        output_back_s, pos_embed_s, inr_embed_s = self.backbone(NestedTensor(search_img, search_att))  # Features & masks, position embedding for the search
        src_feat_s, mask_s = output_back_s[-1].decompose()
        assert mask_s is not None
        # Reduce channel # search region
        feat_s = self.bottleneck(src_feat_s)  # (B, C, H, W)

        # TEMPLATE
        reference_img = data['reference_images'].view(-1, *data['reference_images'].shape[2:])  # (batch, 3, 320, 320)
        reference_att = data['reference_att'].view(-1, *data['reference_att'].shape[2:])  # (batch, 320, 320)
        output_back_t, pos_embed_t, inr_embed_t = self.backbone(NestedTensor(reference_img, reference_att))
        src_feat_t, mask_t = output_back_t[-1].decompose()
        # Reduce channel # search region
        feat_t = self.bottleneck(src_feat_t)
        template_feat = feat_t

        # enhance features by color-depth-aware attention module
        # obtain masks(foreground,  background), histograms(foreground, background) by the template color patch
        search_color = data['search_orimages'].squeeze(0)
        template_color = data['reference_orimages'].squeeze(0)
        target_anno = data['reference_anno'].squeeze(0)*data['search_images'].shape[-1]
        masks, fore_hists_c, back_hists_c = self.generate_template_hist_c(template_color, target_anno)
        # calculate the probability of each pixel of search region belong to foreground, distractors, target
        fore_priors_c = self.generate_search_priors_c(search_color, fore_hists_c)
        back_priors_c = self.generate_search_priors_c(search_color, back_hists_c)
        P_fore_c = fore_priors_c / (fore_priors_c + back_priors_c + 1e-5)
        P_back_c = back_priors_c / (fore_priors_c + back_priors_c + 1e-5)
        priors_c = torch.stack([P_fore_c, P_back_c], dim=1)
        search_feat_c = self.cdaam_color(feat_s, priors_c.cuda())      #enhanced features from color-aware attention

        # obtain masks(foreground,  background), histograms(foreground, background) by the template depth patch
        search_depth = data['search_depths'].squeeze(0)
        template_depth = data['reference_depths'].squeeze(0)
        _, fore_hists_d, back_hists_d = self.generate_template_hist_d(template_depth, target_anno)
        # calculate the probability of each pixel of search region belong to foreground, distractors, target
        fore_priors_d = self.generate_search_priors_d(search_depth, fore_hists_d)
        back_priors_d = self.generate_search_priors_d(search_depth, back_hists_d)

        P_fore_d = fore_priors_d / (fore_priors_d + back_priors_d + 1e-5)
        P_back_d = back_priors_d / (fore_priors_d + back_priors_d + 1e-5)
        priors_d = torch.stack([P_fore_d, P_back_d], dim=1)
        search_feat_d = self.cdaam_depth(feat_s, priors_d.cuda())

        # search_feat_cd = torch.concat([search_feat_c, search_feat_d], dim=1)
        # search_feat = self.conv2d(search_feat_cd)

        search_feat = search_feat_c + search_feat_d + feat_s

        # Adjust shapes  # SEARCH REGION
        feat_vec_s = search_feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_embed_vec_s = pos_embed_s[-1].flatten(2).permute(2, 0, 1)  # HWxBxC
        inr_embed_vec_s = inr_embed_s[-1].flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec_s = mask_s.flatten(1)  # BxHW
        search_dict = {'feat': feat_vec_s, 'mask': mask_vec_s, 'pos': pos_embed_vec_s, 'inr': inr_embed_vec_s}

        # Adjust shapes # template
        feat_vec_t = template_feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_embed_vec_t = pos_embed_t[-1].flatten(2).permute(2, 0, 1)  # HWxBxC
        inr_embed_vec_t = inr_embed_t[-1].flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec_t = mask_t.flatten(1)  # BxHW
        refer_dict = {'feat': feat_vec_t, 'mask': mask_vec_t, 'pos': pos_embed_vec_t, 'inr': inr_embed_vec_t}
        refer_reg = data['reference_region'].squeeze(0)

        return search_dict, refer_dict, refer_reg

    def forward_transformer(self, search_dic, refer_dic=None, refer_reg=None, refer_mem=None,
                            refer_emb=None, refer_pos=None, refer_msk=None):
        if self.aux_loss:
            raise ValueError('ERROR: deep supervision is not supported')

        bs = search_dic['feat'].shape[1]

        # Forward the transformer encoder and decoder
        search_mem = self.transformer.run_encoder(search_dic['feat'], search_dic['mask'], search_dic['pos'],
                                                  search_dic['inr'])

        embed_bank = torch.cat([self.foreground_embed.weight, self.background_embed.weight], dim=0).unsqueeze(0).repeat(
            bs, 1, 1)

        if refer_mem is None:
            refer_mem = self.transformer.run_encoder(refer_dic['feat'], refer_dic['mask'],
                                                     refer_dic['pos'], refer_dic['inr'])
            refer_emb = torch.bmm(refer_reg, embed_bank).transpose(0, 1)

            refer_pos = refer_dic['inr']
            refer_msk = refer_dic['mask']

        output_embed = self.transformer.run_decoder(search_mem, refer_mem, refer_emb, refer_pos, refer_msk)

        return output_embed, search_mem, search_dic['inr'], search_dic['mask']

    def forward_box_head(self, hs):
        """
        Args:
            hs: Output embeddings (1, HW, B, C).
        """

        # Adjust shape
        opt = hs.permute(2, 0, 3, 1).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        # Run the corner head
        bbox_coor = self.box_head(opt_feat)
        coord_in_crop = box_xyxy_to_xywh(bbox_coor)
        outputs_coord = box_xyxy_to_cxcywh(bbox_coor)
        outputs_coord_new = outputs_coord.view(bs, Nq, 4)
        out = {'pred_boxes': outputs_coord_new}

        return out, coord_in_crop #, response_map


    def adjust(self, output_back: list, pos_embed: list, inr_embed: list):
        src_feat, mask = output_back[-1].decompose()
        assert mask is not None
        # Reduce channel
        feat = self.bottleneck(src_feat)  # (B, C, H, W)
        # Adjust shapes
        feat_vec = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_embed_vec = pos_embed[-1].flatten(2).permute(2, 0, 1)  # HWxBxC
        inr_embed_vec = inr_embed[-1].flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec = mask.flatten(1)  # BxHW
        return {'feat': feat_vec, 'mask': mask_vec, 'pos': pos_embed_vec, 'inr': inr_embed_vec}

    def generate_template_hist_c(self, template_color, target_anno):
        '''
        generate template masks by depth values; calculate the histogram of each region;
        input: template_depth: the depth patch of template, target_anno: [x,y,w,h]
        output: histogram of each region: foreground region, background region [B, 3, 256]
                template masks: foreground region, background region [B, W, H]
        '''

        # target_center = torch.round(target_anno[:, 0:2] + target_anno[:, 2:]/2) + 1
        target_bboxes = torch.round(target_anno)
        B, W, H, C = template_color.shape  # B, W, H, C
        fore_masks = torch.zeros([B, W, H])
        back_masks = torch.zeros([B, W, H])
        fore_hists = []
        back_hists = []
        i = 0
        for target_bbox, img_patch in zip(target_bboxes, template_color):
            bbox = target_bbox
            xmin = int(bbox[0])
            xmax = int(bbox[0] + bbox[2])
            ymin = int(bbox[1])
            ymax = int(bbox[1] + bbox[3])
            fore_masks[i][ymin:ymax, xmin:xmax] = 1
            back_masks[i] = 1 - fore_masks[i]

            fore_hists.append(fast_color_histogram(img_patch, 8, fore_masks[i]))
            back_hists.append(fast_color_histogram(img_patch, 8, back_masks[i]))
            i += 1

        masks = torch.stack((fore_masks, back_masks), dim=1)

        return masks, fore_hists, back_hists

    def generate_template_hist_d(self, template_depth, target_anno):
        '''
        generate template masks by depth values; calculate the histogram of each region;
        input: template_depth: the depth patch of template, target_anno: [x,y,w,h]
        output: histogram of each region: foreground region, background region [B, 3, 256]
                template masks: foreground region, background region [B, W, H]
        '''

        # target_center = torch.round(target_anno[:, 0:2] + target_anno[:, 2:]/2) + 1
        target_bboxes = torch.round(target_anno)
        fore_masks = torch.zeros(template_depth.shape)
        back_masks = torch.zeros(template_depth.shape)
        fore_hists = []
        back_hists = []
        i = 0
        for target_bbox, depth_patch in zip(target_bboxes, template_depth):
            bbox = target_bbox
            xmin = int(bbox[0])
            xmax = int(bbox[0] + bbox[2])
            ymin = int(bbox[1])
            ymax = int(bbox[1] + bbox[3])
            fore_masks[i][ymin:ymax, xmin:xmax] = 1
            back_masks[i] = 1 - fore_masks[i]
            fore_hists.append(
                cv2.calcHist([np.array(depth_patch.cpu())], [0], np.uint8(fore_masks[i]), [250], [0, 25000]) / (
                            np.sum(np.uint8(fore_masks[i])) + 1))
            back_hists.append(
                cv2.calcHist([np.array(depth_patch.cpu())], [0], np.uint8(back_masks[i]), [250], [0, 25000]) / (
                            np.sum(np.uint8(back_masks[i])) + 1))
            i += 1

        masks = torch.stack((fore_masks, back_masks), dim=1)

        return masks, fore_hists, back_hists

    def generate_search_priors_c(self, search_color, hists):

        hists = torch.stack([torch.tensor(hist) for hist in hists])
        B, W, H, C = search_color.shape
        prior_maps = torch.zeros([B, W, H])
        image_index = torch.tensor(search_color, dtype=torch.long) // 8

        for i in range(B):
            index = image_index[i].reshape(-1, C).T
            prior_maps[i] = hists[i][index[0], index[1], index[2]].reshape(W, H)
        return prior_maps

    def generate_search_priors_d(self, search_depth, hists):

        hists = torch.stack([torch.from_numpy(hist) for hist in hists])
        B, W, H = search_depth.shape
        search_depth_src = search_depth.view(B, -1)
        search_bin_indices = torch.floor(search_depth_src/100)
        prior_maps = torch.zeros([B, W, H])
        for i in range(B):
            prior_maps[i] = hists[i][np.array(search_bin_indices[i].cpu())].reshape(W, H)

        return prior_maps

    def template(self, input: NestedTensor, color_patch,  depth_patch, bbox):

        assert isinstance(input, NestedTensor)
        output_back, pos_embed, inr_embed = self.backbone(input)  # Features & masks, position embedding for the search
        src_feat, mask = output_back[-1].decompose()
        # Reduce channel # search region
        feat = self.bottleneck(src_feat)
        anno_bbox = torch.from_numpy(np.expand_dims(bbox, axis=0))
        masks, fore_hists_c, back_hists_c = self.generate_template_hist_c(torch.from_numpy(color_patch.astype(np.float32)).unsqueeze(0), anno_bbox)
        _, fore_hists_d, back_hists_d = self.generate_template_hist_d(torch.from_numpy(depth_patch.astype(np.float32)).unsqueeze(0), anno_bbox)

        # save histograms for online update
        self.fore_hists_template_c = fore_hists_c
        self.back_hists_template_c = back_hists_c
        self.fore_hists_template_d = fore_hists_d
        self.back_hists_template_d = back_hists_d

        template_feat = feat
        # Adjust shapes # template
        feat_vec_t = template_feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_embed_vec_t = pos_embed[-1].flatten(2).permute(2, 0, 1)  # HWxBxC
        inr_embed_vec_t = inr_embed[-1].flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec_t = mask.flatten(1)  # BxHW
        refer_dict = {'feat': feat_vec_t, 'mask': mask_vec_t, 'pos': pos_embed_vec_t, 'inr': inr_embed_vec_t}

        return refer_dict

    def search(self, input: NestedTensor, color_patch, depth_patch):

        assert isinstance(input, NestedTensor)
        output_back, pos_embed, inr_embed = self.backbone(input)  # Features & masks, position embedding for the search
        src_feat, mask = output_back[-1].decompose()
        # Reduce channel # search region
        feat = self.bottleneck(src_feat)

        # get color-aware attention map
        color_patch = torch.from_numpy(color_patch.astype(np.float32)).unsqueeze(0)
        # calculate the probability of each pixel of search region belong to foreground, distractors, target
        fore_priors_c = self.generate_search_priors_c(color_patch, self.fore_hists_template_c)
        back_priors_c = self.generate_search_priors_c(color_patch, self.back_hists_template_c)

        P_fore_c = fore_priors_c / (fore_priors_c + back_priors_c + 1e-5)
        P_back_c = back_priors_c / (fore_priors_c + back_priors_c + 1e-5)
        priors_c = torch.stack([P_fore_c, P_back_c], dim=1)
        search_feat_c = self.cdaam_color(feat, priors_c.cuda())

        # get color-aware attention map
        depth_patch = torch.from_numpy(depth_patch.astype(np.float32)).unsqueeze(0)
        # calculate the probability of each pixel of search region belong to foreground, distractors, target
        fore_priors_d = self.generate_search_priors_d(depth_patch, self.fore_hists_template_d)
        back_priors_d = self.generate_search_priors_d(depth_patch, self.back_hists_template_d)

        P_fore_d = fore_priors_d / (fore_priors_d + back_priors_d + 1e-5)
        P_back_d = back_priors_d / (fore_priors_d + back_priors_d + 1e-5)
        priors_d = torch.stack([P_fore_d, P_back_d], dim=1)
        search_feat_d = self.cdaam_depth(feat, priors_d.cuda())

        # fuse
        search_feat = search_feat_c + search_feat_d + feat

        # Adjust shapes # template
        feat_vec_s = search_feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_embed_vec_s = pos_embed[-1].flatten(2).permute(2, 0, 1)  # HWxBxC
        inr_embed_vec_s = inr_embed[-1].flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec_s = mask.flatten(1)  # BxHW
        search_dict = {'feat': feat_vec_s, 'mask': mask_vec_s, 'pos': pos_embed_vec_s, 'inr': inr_embed_vec_s}

        return search_dict

    def update_hists_d(self, depth_patch, anno_bbox):
        update_lr = 0.1
        masks, fore_hists_d, back_hists_d = self.generate_template_hist_d(
            torch.from_numpy(depth_patch.astype(np.float32)).unsqueeze(0), anno_bbox)

        self.fore_hists_template_d = [self.fore_hists_template_d[0] * (1-update_lr) + fore_hists_d[0] * update_lr]
        self.back_hists_template_d = [self.back_hists_template_d[0] * (1-update_lr) + back_hists_d[0] * update_lr]


def build_cdaatrack(cfg):
    backbone = build_backbone(cfg)  # Backbone and positional encoding are built together
    transformer = build_transformer(cfg)
    box_head = build_box_head(cfg)
    model = CDAATRACK(
        backbone,
        transformer,
        box_head,
        num_queries=cfg.MODEL.NUM_OBJECT_QUERIES,
        aux_loss=cfg.TRAIN.DEEP_SUPERVISION,
        head_type=cfg.MODEL.HEAD_TYPE,
    )

    # load the weights trained on RGB datasets
    if os.path.isfile(cfg.MODEL.PRETRAINED):
        ckpt = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(ckpt['net'], strict=False)
        print("missing keys:", missing_keys)
        print("unexpected keys:", unexpected_keys)
        print("Loading pretrained RGB-only baseline weights done.")
    return model
