import argparse
import os
import torch


class DataOptions():
    """
    This class defines options used during both training and test time.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initialized"""
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--phase', type=str, default='train', help='train or evaluation (isotropic volume export)')
        parser.add_argument('--csv_path', required=True, default='your_csv_file.csv', help='path to the dataset manifest CSV')
        parser.add_argument('--patch_size', required=True, type=int, default=512, help='spatial size (H and W) of training/inference patches')
        parser.add_argument('--planes', required=True, type=str, default='coronal,axial', help='the planes to train e.g. coronal  coronal,axial,sagittal')
        parser.add_argument('--main_dir', required=True, default='outputs', help='path to save script outputs')

        parser.add_argument(
            '--intensity_norm_mode',
            type=str,
            default='minmax',
            choices=['stretch', 'clip', 'fixed_range', 'minmax'],
            help=(
                'How to normalize voxel intensities to [-1, 1]. '
                '"stretch"     – auto-compute per-volume bounds (percentiles) then linearly stretch the top intensities into the full range. '
                '"clip"        – clip each volume to its per-case [intensity_min, intensity_max] (read from CSV), then map linearly to [-1, 1]. '
                '"fixed_range" – clip every volume in the cohort to the same [intensity_min, intensity_max] (from --intensity_min / --intensity_max), then map linearly to [-1, 1]. '
                '"minmax"      – no clipping; pure per-volume min-max rescale to [-1, 1].'
            ),
        )
        parser.add_argument(
            '--intensity_min',
            type=float,
            default=0.0,
            help=(
                'Lower intensity bound. '
                'Used as the cohort-wide lower clip value in "fixed_range" mode, '
                'or as a fallback lower bound in "clip" mode when the CSV row has no intensity_min.'
            ),
        )
        parser.add_argument(
            '--intensity_max',
            type=float,
            default=2048.0,
            help=(
                'Upper intensity bound. '
                'Used as the cohort-wide upper clip value in "fixed_range" mode, '
                'or as a fallback upper bound in "clip" mode when the CSV row has no intensity_max.'
            ),
        )
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        # parser.add_argument('--data_save_dir', type=str, default=None, help='path to save data')
        parser.add_argument('--use_only_train', action='store_true', default=False, help='use all data for training without a validation split')
        parser.add_argument('--default_plane', type=str, default='coronal', help='default plane to create isotropic volume')
        parser.add_argument(
            '--build_slice_cache',
            action='store_true',
            default=False,
            help='preprocess volumes into per-slice .pt caches under <main_dir>/CRIS_Dataset/',
        )
        parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
        parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
        parser.add_argument(
            '--export_isotropic_volumes',
            action='store_true',
            default=False,
            help='after training, run isotropic volume export on the test split (same as --phase evaluation)',
        )
        parser.add_argument('--pre_trained', type=str, default=None, help='pre trained model path')
        parser.add_argument(
            '--dataset_path',
            type=str,
            default=None,
            help=(
                'root directory of preprocessed slice data (<plane>/train|val). '
                'Defaults to <main_dir>/CRIS_Dataset when unset.'
            ),
        )
        parser.add_argument('--gap', type=int, default=5, help='gap')
        parser.add_argument(
            '--no_degradation',
            action='store_true',
            default=False,
            help=(
                'Skip the in-plane 1-D blur that simulates thick-slice acquisition physics (sets sigma=0). '
                'When enabled, known slices are passed to the network as-is, with no blurring applied. '
                'This can yield the best-looking results when evaluating without ground-truth (GT), '
                'because the input slices are already high-resolution and no artificial blur is introduced. '
                'WARNING: Do NOT use this flag if you are computing quantitative metrics against a GT volume. '
                'The model was trained with the degradation, so disabling it at inference creates a '
                'train/test mismatch that will hurt PSNR/SSIM scores.'
            ),
        )

        ########## Model specific options ##########
        parser.add_argument('--n_epochs', type=int, default=150, help='number of epochs to train the model')
        parser.add_argument('--save_epoch_freq', type=int, default=50, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--base_filters', type=int, default=96, help='base filters for the model')
        parser.add_argument('--window_size', type=int, default=8, help='Swin bottleneck window size')
        parser.add_argument('--output_nc', type=int, default=1, help='output channels')
        parser.add_argument('--domain', type=str, default='MRI', help='domain of the data, e.g. MRI, microscopy')

        self.initialized = True
        return parser

    def gather_options(self, parser=None):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            if parser is None:
                parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return opt

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        os.makedirs(opt.main_dir, exist_ok=True)
        file_name = os.path.join(opt.main_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, parser=None):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options(parser)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])
            opt.device = torch.device('cuda:{}'.format(opt.gpu_ids[0]))
        else:
            opt.device = torch.device('cpu')

        # set planes
        opt.planes = opt.planes.split(',')

        if opt.dataset_path is None:
            opt.dataset_path = os.path.join(opt.main_dir, 'CRIS_Dataset')

        self.opt = opt

        return self.opt
