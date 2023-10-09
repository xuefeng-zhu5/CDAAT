import torch

from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from lib.utils.merge import merge_feature_sequence
from lib.utils.misc import NestedTensor
from . import BaseActor


class CDAATRACKActor(BaseActor):
    """
    Actor for training.
    """

    def __init__(self, net, objective, loss_weight, settings):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # Batch size

    def __call__(self, data):
        """
        Args:
            data: The input data, should contain the fields 'reference', 'search', 'gt_bbox'.
            reference_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)

        Returns:
            loss: The training loss.
            status: Dict containing detailed losses.
        """

        # Forward pass
        out_dict = self.forward_pass(data)

        # Compute losses
        # Process the groundtruth
        gt_bboxes = data['search_anno']  # (Ns, batch, 4) (x1,y1,w,h)
        loss, status = self.compute_losses(out_dict, gt_bboxes[0], data['proposal_iou'])
        return loss, status

    def forward_pass(self, data):
        # # Process the search regions (t-th frame)

        # extract CNN features & DAA
        search_dict, feat_dict, refer_reg = self.net(data, mode='backbone')

        # Run the transformer and compute losses  # only one template
        out_embed, _, _, _ = self.net(search_dic=search_dict, refer_dic=feat_dict,
                                      refer_reg=refer_reg, mode='transformer')

        # Forward the corner head
        out_dict, _ = self.net(out_embed=out_embed, mode='head')  # out_embed: [1, 400, 8, 256], out_dict: (B, N, C), outputs_coord: (1, B, N, C)

        return out_dict

    def compute_losses(self, pred_dict, gt_bbox, iou_gt, return_status=True):
        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError('ERROR: network outputs is NaN! stop training')
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # Compute GIoU and IoU
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # Compute L1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)

        # iou_pred = pred_dict['pred_iou']
        # iou_loss = self.objective['iou'](iou_pred, iou_gt)

        # Weighted sum
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
        #     'iou'] * iou_loss
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss
        if return_status:
            # Status for log
            mean_iou = iou.detach().mean()
            status = {'Ls/total': loss.item(),
                      'Ls/giou': giou_loss.item(),
                      'Ls/l1': l1_loss.item(),
                      'IoU': mean_iou.item()}
            return loss, status
        else:
            return loss
