from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = ''
    settings.got10k_path = '/vol/research/facer2vm_tracking/Datasets/GOT_10k/got_10k_data'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = ''
    settings.lasot_path = ''
    settings.network_path = '/vol/research/facer2vm_tracking/people/xuefeng/CDAAT/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.prj_dir = '/vol/research/facer2vm_tracking/people/xuefeng/CDAAT'
    settings.result_plot_path = '/vol/research/facer2vm_tracking/people/xuefeng/CDAAT/test/result_plots'
    settings.results_path = '/vol/research/facer2vm_tracking/people/xuefeng/CDAAT/tracking_results' # Where to store tracking results
    settings.save_dir = '/vol/research/facer2vm_tracking/people/xuefeng/CDAAT/lib/train'
    settings.segmentation_path = '/vol/research/facer2vm_tracking/people/xuefeng/CDAAT/test/segmentation_results'
    settings.tc128_path = '/vol/research/facer2vm_tracking/people/xuefeng/CDAAT/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''
    settings.depthtrack_path = '/vol/research/facer2vm_tracking/people/xuefeng/VOT_workspace/workspace22/sequences_depthtrack'
    settings.cdtb_path = '/vol/research/facer2vm_tracking/people/xuefeng/VOT_workspace/workspace22/sequences_cdtb'
    settings.rgbd1k_path = '/vol/research/facer2vm_tracking/people/xuefeng/VOT_workspace/workspace22/sequences_rgbd1k'

    return settings

