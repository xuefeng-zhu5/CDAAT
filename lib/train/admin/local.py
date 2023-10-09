class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/vol/research/facer2vm_tracking/people/xuefeng/CDAAT'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks'

        self.lasot_dir = '/vol/research/facer2vm_tracking/Datasets/LaSOTBenchmark'
        self.got10k_dir = '/vol/research/facer2vm_tracking/Datasets/GOT_10k/got_10k_data/train'
        self.trackingnet_dir = '/vol/research/facer2vm_tracking/Datasets/TrackingNet'
        self.coco_dir = '/vol/research/facer2vm_tracking/Datasets/COCO'

        self.rgbd1k_dir = '/vol/research/facer2vm_tracking/people/xuefeng/Benchmarks/RGBD1K_train_labelled'
        self.depthtrack_dir = '/vol/research/facer2vm_tracking/people/xuefeng/Benchmarks/DepthTrack/Train'


