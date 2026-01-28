from train import *
from eval.utils import run_inference_on_folder

import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir',
    type = str,
    help = 'Path to data. Inside it should contain base images for generating defects',
    required=True
)
parser.add_argument(
    '--model_dir',
    type = str,
    help = 'Path to abnormal image generator',
    required=True
)
parser.add_argument(
    '--out_dir',
    type=str,
    help='Where to save generated images (default: <model_dir>/generated)',
    default=None
)
args = parser.parse_args()


def main():
    base_dir = args.data_dir
    model_dir = args.model_dir
    out_dir = args.out_dir

    num_classes = 3
    class_map = {1: "outer", 2: "inner"}

    run_inference_on_folder(
        checkpoint_path=model_dir,
        images_dir= base_dir,
        output_dir = out_dir,
        num_classes=num_classes,
        score_thresh = 0.2,
        class_map=class_map,
        use_pretrained = False,
        save_crops=True
    )
if __name__ == '__main__':
    main()