import torch
from torch import nn

from lib.utils.box_ops import box_xyxy_to_cxcywh, box_xyxy_to_xywh
from lib.utils.misc import NestedTensor
from .backbone import build_backbone
from .head import build_box_head
from .transformer import build_transformer


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


class BASELINE(BASETRACK):
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


    def forward(self, img=None, search_dic=None, refer_dic_list=None, refer_reg_list=None, out_embed=None,
                mode='backbone'):
        if mode == 'backbone':
            return self.forward_backbone(img)
        elif mode == 'transformer':
            return self.forward_transformer(search_dic, refer_dic_list, refer_reg_list)
        elif mode == 'head':
            return self.forward_box_head(out_embed)
        else:
            raise ValueError

    def forward_backbone(self, input: NestedTensor):
        """
        The input type is NestedTensor, which consists of:
            tensor: Batched images, of shape [batch_size x 3 x H x W].
            mask: A binary mask of shape [batch_size x H x W], containing 1 on padded pixels.
        """

        assert isinstance(input, NestedTensor)
        # Forward the backbone
        output_back, pos, inr = self.backbone(input)  # Features & masks, position embedding for the search
        # Adjust the shapes
        return self.adjust(output_back, pos, inr)

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
        return out, coord_in_crop


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


def build_baseline(cfg):
    backbone = build_backbone(cfg)  # Backbone and positional encoding are built together
    transformer = build_transformer(cfg)
    box_head = build_box_head(cfg)
    model = BASELINE(
        backbone,
        transformer,
        box_head,
        num_queries=cfg.MODEL.NUM_OBJECT_QUERIES,
        aux_loss=cfg.TRAIN.DEEP_SUPERVISION,
        head_type=cfg.MODEL.HEAD_TYPE,
    )

    return model
