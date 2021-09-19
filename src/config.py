import argparse
import os.path as op


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args_function():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainsplit",
        type=str,
        default='trainval',
        choices=['train', 'minitrain', 'trainval'],
        help="You should use train by default but use minitrain to speed up coding")
    parser.add_argument(
        "--valsplit",
        type=str,
        default='val',
        choices=['val', 'minival'],
        help="You should use val by default but use minival to speed up coding")
    parser.add_argument(
        "--semisplit",
        type=str,
        default='full',
        choices=['full', 'mini'],
        help="Unpaired data split used. Use mini to speed up coding")
    parser.add_argument(
        "--input_img_size",
        type=int,
        default=128,
        help="Input image size. Do not change this from its default"
    )
    parser.add_argument(
        "--load_ckpt",
        type=str,
        default="",
        help='Load pre-trained weights from your training procedure for test.py. e.g., logs/EXP_KEY/latest.pt'
    )
    parser.add_argument(
        "--final",
        type=int,
        default=55,
        help="Number of epochs to train in the final phase"
    )
    parser.add_argument(
        "--final_splitted",
        type=int,
        default=65,
        help="Number of epochs to train after first split"
    )
    parser.add_argument(
        "--init_splitted",
        type=int,
        default=5,
        help="Number of epochs to init after first split"
    )
    parser.add_argument(
        "--eval_every_epoch",
        type=int,
        default=1,
        help="Evaluate your model in the training process every K training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2.5*1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="",
        help="Path where to save model weights"
    )
    parser.add_argument(
        "--segmentation_model_g1",
        type=str,
        default="uxnet",
        help="Segmentation model to use to pretrain"
    )
    parser.add_argument(
        "--segmentation_model_g2",
        type=str,
        default="uplusregnet",
        help="Segmentation model to use to train"
    )
    parser.add_argument(
        "--datasets",
        type=int,
        default=3,
        help="How many datasets to use (1 = paired, 2 = paired + unp_labels 3 = paired + unp_labels + unp_img"
    )
    parser.add_argument(
        "--pre",
        type=int,
        default=100,
        help="How much training for G1"
    )
    parser.add_argument(
        "--init",
        type=int,
        default=15,
        help="How much pretraining for G2 on fake pairs produced by G1"
    )
    parser.add_argument(
        "--saved_weights_path_g1",
        type=str,
        default=None,
        help="Saved weights (state_dict) path for generator g1"
    )
    parser.add_argument(
        "--segm_loss_w",
        type=float,
        default=0.35,
        help="weight for segm loss (cross entropy)"
    )
    parser.add_argument(
        "--main_loss_w",
        type=float,
        default=0.65,
        help="weight for dice loss"
    )
    parser.add_argument(
        "--discriminator_w",
        type=float,
        default=0.1,
        help="weight for discriminator loss"
    )
    parser.add_argument(
        "--saved_weights_path_D",
        type=str,
        default=None,
        help="Saved weights (state_dict) path for discriminator"
    )
    parser.add_argument(
        "--comet_name",
        type=str,
        default="experiment",
        help="name of the experiment to display on comet"
    )
    parser.add_argument(
        "--use_comet",
        type=str2bool,
        default=False,
        help="whether to use comet or not"
    )
    parser.add_argument(
        "--use_weights",
        type=str2bool,
        default=True,
        help="whether to use ce weights or not"
    )
    parser.add_argument(
        "--use_old_weights",
        type=str2bool,
        default=True,
        help="whether to use old weights or new ones"
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="dice",
        help="name of the loss to use"
    )
    parser.add_argument(
        "--dryrun",
        type=str2bool,
        default=False,
        help="dry run to check that the model is working"
    )

    args = parser.parse_args()

    root_dir = op.join('.')
    data_dir = op.join(root_dir, 'data')
    args.input_img_shape = tuple([args.input_img_size]*2)
    args.root_dir = root_dir
    args.data_dir = data_dir
    args.experiment = None
    return args


args = parse_args_function()
