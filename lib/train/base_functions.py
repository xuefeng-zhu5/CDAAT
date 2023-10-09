import torch
from torch.utils.data.distributed import DistributedSampler

import lib.train.data.transforms as tfm
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
# Datasets related
from lib.train.dataset import Lasot, Got10k, MSCOCOSeq, TrackingNet, RGBD1K, DepthTrack


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'search': cfg.DATA.SEARCH.FACTOR,
                                   'reference': cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {'search': cfg.DATA.SEARCH.SIZE,
                          'reference': cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {'search': cfg.DATA.SEARCH.CENTER_JITTER,
                                     'reference': cfg.DATA.SEARCH.CENTER_JITTER,
                                     'initial': cfg.DATA.TEMPLATE.CENTER_JITTER}
    settings.scale_jitter_factor = {'search': cfg.DATA.SEARCH.SCALE_JITTER,
                                    'reference': cfg.DATA.SEARCH.SCALE_JITTER,
                                    'initial': cfg.DATA.TEMPLATE.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE

def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in ["Depthtrack_train", "RGBD1K", "Depthtrack_val", "LASOT", "GOT10K_vottrain", "GOT10K_votval",
                        "GOT10K_train_full", "COCO17", "TRACKINGNET", "Depthtrack_train_RGB", "Depthtrack_val_RGB", "RGBD1K_RGB"]
        if name == "Depthtrack_train":
            datasets.append(DepthTrack(settings.env.depthtrack_dir, dtype='rgbrawd', split='train', image_loader=image_loader))
        if name == "Depthtrack_val":
            datasets.append(DepthTrack(settings.env.depthtrack_dir, dtype='rgbrawd', split='val', image_loader=image_loader))
        if name == "RGBD1K":
            datasets.append(RGBD1K(settings.env.rgbd1k_dir, dtype='rgbrawd', image_loader=image_loader))

        if name == "Depthtrack_train_RGB":
            datasets.append(DepthTrack(settings.env.depthtrack_dir, dtype='color', split='train', image_loader=image_loader))
        if name == "Depthtrack_val_RGB":
            datasets.append(DepthTrack(settings.env.depthtrack_dir, dtype='color', split='val', image_loader=image_loader))
        if name == "RGBD1K_RGB":
            datasets.append(RGBD1K(settings.env.rgbd1k_dir, dtype='color', image_loader=image_loader))
        if name == "LASOT":
            datasets.append(Lasot(settings.env.lasot_dir, split='train', image_loader=image_loader))
        if name == "GOT10K_vottrain":
            datasets.append(Got10k(settings.env.got10k_dir, split='vottrain', image_loader=image_loader))
        if name == "GOT10K_train_full":
            datasets.append(Got10k(settings.env.got10k_dir, split='train_full', image_loader=image_loader))
        if name == "GOT10K_votval":
            datasets.append(Got10k(settings.env.got10k_dir, split='votval', image_loader=image_loader))
        if name == "COCO17":
            datasets.append(MSCOCOSeq(settings.env.coco_dir, version="2017", image_loader=image_loader))
        if name == "TRACKINGNET":
            datasets.append(TrackingNet(settings.env.trackingnet_dir, image_loader=image_loader))

    return datasets


def build_dataloaders(cfg, settings):
    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))



    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    if settings.config_name == 'baseline':
        ProcessingClass= processing.BASELINEProcessing
        SamplerClass = sampler.BASELINESampler
    else:
        ProcessingClass= processing.CDAATRACKProcessing
        SamplerClass = sampler.CDAATSampler

    data_processing_train = ProcessingClass(search_area_factor=search_area_factor,
                                                          output_sz=output_sz,
                                                          center_jitter_factor=settings.center_jitter_factor,
                                                          scale_jitter_factor=settings.scale_jitter_factor,
                                                          mode='sequence',
                                                          transform=transform_train,
                                                          joint_transform=transform_joint,
                                                          settings=settings)

    # Train sampler and loader
    dataset_train = SamplerClass(
        datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
        p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
        samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
        max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
        processing=data_processing_train)

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)


    # validation
    data_processing_val = ProcessingClass(search_area_factor=search_area_factor,
                                                          output_sz=output_sz,
                                                          center_jitter_factor=settings.center_jitter_factor,
                                                          scale_jitter_factor=settings.scale_jitter_factor,
                                                          mode='sequence',
                                                          transform=transform_val,
                                                          joint_transform=transform_joint,
                                                          settings=settings)
    dataset_val = SamplerClass(
        datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader),
        p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
        samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
        max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
        processing=data_processing_val)
    val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                           num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
                           epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)


    return loader_train, loader_val


def get_optimizer_scheduler(net, cfg):
    param_dicts = [
        {'params': [p for n, p in net.named_parameters() if 'backbone' not in n and p.requires_grad]},
        {'params': [p for n, p in net.named_parameters() if 'backbone' in n and p.requires_grad],
         'lr': cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER}
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)

    return optimizer, lr_scheduler
