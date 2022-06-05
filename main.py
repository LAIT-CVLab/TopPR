from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
import os
from utils.util import make_main_results_dirs


def args_parser(ipynb=False):
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    #########################################################
    #   Config for Extracting Feature Vectors
    #########################################################
    parser.add_argument('--path_real', type=str, default='./example',
                        help='Path to the real images')
    parser.add_argument('--path_fake', type=str, default='./example',
                        help='Path to the fake images')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size to use')
    parser.add_argument('--generated_image_size', type=int, default=64,
                        help='resolution of generated images, 32 for cifar-10, 64 for lsun-bedroom and celeba')
    parser.add_argument('--image_size', type=int, default=224,
                        help='224 for vgg16 and resnet, 299 for inception')
    parser.add_argument('--nearest_k', type=int, default=5,
                        help='k value to use')
    parser.add_argument('--num_samples', type=int, default=-1,
                        help='number of samples to use')
    parser.add_argument('--seed', type=int, default=0, help='Seed value')
    parser.add_argument("--pretrained", type=bool, nargs='?', const=True,
                        default=True, help="Use a pretrained network.")
    parser.add_argument('--arch', type=str, default='vgg16',
                        help='Embedding model, please look in get_features to ')
    parser.add_argument('--feature', type=int, default=None,
                        help='Size of feature dimension (currently only vgg11, '
                             'vgg16, vgg19, renst50)')


    #########################################################
    #   Config for Experiment setting
    #########################################################
    parser.add_argument('--truncation_trick_toy', type=bool, default=False, help='')
    parser.add_argument('--truncation_trick_real_data', type=bool, default=False, help='')
    parser.add_argument('--simultaneous_mode_drop_toy', type=bool, default=False, help='')
    parser.add_argument('--simultaneous_mode_drop_real_data', type=bool, default=False, help='')
    parser.add_argument('--sequential_mode_drop_toy', type=bool, default=False, help='')
    parser.add_argument('--sequential_mode_drop_real_data', type=bool, default=False, help='')
    parser.add_argument('--inoutlier_toy', type=bool, default=False, help='')
    parser.add_argument('--inoutlier_real_data', type=bool, default=False, help='')
    parser.add_argument('--realism_score_real_data', type=bool, default=False, help='')
    parser.add_argument('--single_score', type=bool, default=False, help='')
    parser.add_argument('--per_sample_score', type=bool, default=False, help='')
    parser.add_argument('--check_outlier', type=bool, default=False, help='')


    #########################################################
    #   Config for Metric setting
    #########################################################
    parser.add_argument('--isscore', type=bool, default=False, help='')
    parser.add_argument('--fid', type=bool, default=False, help='')
    parser.add_argument('--prdc', type=bool, default=False, help='')
    parser.add_argument('--lpips', type=bool, default=False, help='')
    parser.add_argument('--tprdc', type=bool, default=False, help='')

    #########################################################
    #   Config for Saving Datas
    #########################################################
    parser.add_argument('--outf', type=str, default='.', required=True, help='folder name for saving main_path')


    if ipynb:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    return args



def main():
    args = args_parser(ipynb=True)
    make_main_results_dirs(args.outf, args)
    

if __name__ == '__main__':
    main()