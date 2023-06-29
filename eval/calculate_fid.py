import argparse
import os
from pathlib import Path
from typing import Callable, Optional
import pandas as pd

from cleanfid import fid

import torch

from accelerate.utils import set_seed

import logging

from torchvision import transforms
from tqdm import tqdm
from pprint import pprint

import numpy as np

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s; %(levelname)s: %(message)s')
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)

def parse_args():
    parser = argparse.ArgumentParser(description="")

    # Required params
    parser.add_argument(
        "primary_input",
        type=str,
        help="Folder or csv-file to load instances from"
    )
    parser.add_argument(
        "secondary_input",
        type=str,
        help="Second folder or csv file to load instances from",
    )

    # General options
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="experiment",
        help="Experiment name"
    )   
    parser.add_argument(
        "--fast_dev_run",
        default=False,
        action="store_true",
        help="Fast development run to test the script on a maximum of 5 images"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cached_fid_feats",
        help="Directory where features are cached"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "The output directory where the calculations are stored. " \
            "If not provided, outputs are written to stdout"
        )
    )
    parser.add_argument(
        "--n_images",
        type=int,
        default=5000,
        help="Number of images to use for computation"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the processing dataloader"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible results in case randomization is used"
    )
    parser.add_argument(
        "--suppress_verbose",
        default=False,
        action="store_true",
        help="If verbose information about the computations should be displayed"
    )

    # Feature extractors
    parser.add_argument(
        "--inception_fid",
        type=int,
        default=None,
        choices=[2048], # [64,128,768,2048] # TODO: Implement
        help="Use ImageNet-pretrained InceptionV3 as feature extractor (specify layer, default 2048)"
    )
    parser.add_argument(
        "--xrv_fid",
        default=False,
        action="store_true",
        help="Use torchxrayvision's DenseNet-121 pretrained on CXR (weights==all) as feature extractor (1024 layer)"
    )
    parser.add_argument(
        "--clip_fid",
        default=False,
        action="store_true",
        help="Use CLIP as feature extractor"
    )

    # Distributed
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers"
    )

    # File list creation
    parser.add_argument(
        "--search_pattern",
        type=str,
        default="*_0.jpg",
        help="Search pattern for filenames."
    )

    parser.add_argument(
        "--match_file_indices",
        default=False,
        action="store_true",
        help="If True, the file ids (e.g., MIMIC ids) in the primary and secondary input are matched."
    )

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    if not os.path.exists(args.primary_input):
        raise FileNotFoundError(f"Input folder or .csv file not found ({args.primary_input})")
    if not os.path.exists(args.secondary_input):
        raise FileNotFoundError(f"Input folder or .csv file not found ({args.secondary_input})")
    return args



def main():
    """Calculate the FID between two folders using different feature extractors

    Example usage:
    python3 calculate_fid.py \
        --primary_input /path/to/folder1 \
        --secondary_input /path/to/folder2 \
        --n_images 1000 \
        --inception_fid 2048 \
        --xrv_fid \
        --clip_fid \
        --output_dir /path/to/output_dir \
        --experiment_name "my_experiment" \
        --seed 42 \
        --batch_size 32 \
        --num_workers 4 \
        
    Args:
        primary_input (str): Path to folder or csv-file to load instances from
        secondary_input (str): Second folder or csv file to load instances from
        n_images (int): Number of images to use for computation
        inception_fid (int): Use ImageNet-pretrained InceptionV3 as feature extractor (specify layer, default 2048)
        xrv_fid (bool): Use torchxrayvision's DenseNet-121 pretrained on CXR (weights==all) as feature extractor (1024 layer)
        clip_fid (bool): Use CLIP as feature extractor
        output_dir (str): The output directory where the calculations are stored. If not provided, outputs are written to stdout
        experiment_name (str): Experiment name
        seed (int): A seed for reproducible results in case randomization is used
        batch_size (int): Batch size (per device) for the processing dataloader
        num_workers (int): Number of workers

    """
    args = parse_args()
    verbose = not args.suppress_verbose
    if not verbose:
        logger.debug(args)

    n_images = 5 if args.fast_dev_run else args.n_images
    logger.info(f"Using {n_images} images.")

    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
      if not os.path.exists(args.output_dir):
        logger.warn(f"The specified output directory doesn't exist. ({args.output_dir})")
        output_dir = None
      else:
        output_dir = args.output_dir

    results = {}
    results["args"] = dict(args.__dict__)

    if args.device is not None:
        device = args.device
    else:
        logger.info("Detecting device.")
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    results["device"] = device
    logger.info(f"Using device {device}")

    # Get file lists
    # TODO add support for custom file lists
    # TODO add support for base_dir
    pattern = args.search_pattern # defaults to "*_0.jpg" currently

    # flist1 holds the first folder
    # flist2 holds the second folder
    def find_files(input_dir, pattern):
        if os.path.isdir(input_dir):
            logger.info(f"Looking for image files in {input_dir}...")
            logger.info(f"Search pattern: {pattern}")
            return list(tqdm(Path(input_dir).glob(pattern)))
        elif str(input_dir).endswith(".csv"):
            df1 = pd.read_csv(input_dir)
            assert "Path" in df1.columns, ".csv was loaded but didn't contain 'Path' column"
            return list(df1.Path.values)

    flist1 = find_files(args.primary_input, pattern)
    logger.info(f"Found {len(flist1)} files in {args.primary_input}")

    flist2 = find_files(args.secondary_input, "*.jpg")
    logger.info(f"Found {len(flist2)} files in {args.secondary_input}")

    n_images = 5 if args.fast_dev_run else args.n_images
    if args.n_images is not None:
        # Take whatever is smaller, n_images or all files in the flist2
        flist2 = flist2[:np.min([len(flist2),n_images])]
        if args.match_file_indices:
            flist1 = [os.path.join(args.primary_input,x.as_posix().split("/")[-1].replace(".jpg","_0.jpg"))
                    for x in flist2]
            logger.info("!!! Replacing flist 1 with matched IDs in this setup !!!")

            # Check that IDs match
            ids1 = [f.split("/")[-1].replace("_0.jpg","") for f in flist1]
            ids2 = [f.as_posix().split("/")[-1].replace(".jpg","") for f in flist2]
            assert set(ids1) == set(ids2), "Ids dont match"
        else:
            flist1 = flist1[:np.min([len(flist1),n_images])]

        logger.info(f"Last image flist1: {flist1[-1]}")
        logger.info(f"Last image flist2: {flist2[-1]}")

    def calc_feats_from_flist(
        flist,
        feat_model,
        custom_fn_resize=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        verbose=verbose
    ):
        return fid.get_files_features(
            l_files=flist,
            model=feat_model,
            mode="clean",
            custom_fn_resize=custom_fn_resize,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            verbose=verbose
        )

    def create_cached_label(s):
        """Utility function to create cache label from args.*_input"""
        return s.replace(" ","_").replace("/","_")

    def cache_files(flist, input_dir, method_prefix, feat_model, custom_fn_resize=None, cache_dir=args.cache_dir):
        cache_file = Path(cache_dir)/create_cached_label(f"{method_prefix}_{input_dir}_n={len(flist)}.npz")
        if os.path.exists(cache_file):
            logger.info(f"Loaded cached features: {cache_file}")
            feats = np.load(cache_file)["feats"]
        else:
            logger.info(f"Calculating features for {input_dir}")
            feats = calc_feats_from_flist(flist, feat_model, custom_fn_resize=custom_fn_resize)
            np.savez_compressed(cache_file, feats=feats)
        return feats

    ###
    #  Inception V3 (default FID)
    ###
    if args.inception_fid is not None:
        logger.info("Computing FID using InceptionV3")
        logger.info("NB: for now, only the 2048 layer is supported")

        feat_model = fid.build_feature_extractor(
            mode="clean",
            device=torch.device(device),
            use_dataparallel=True
        )
        
        feats1 = cache_files(flist1, args.primary_input, "inception", feat_model)
        feats2 = cache_files(flist2, args.secondary_input,  "inception", feat_model)

        score = fid.fid_from_feats(feats1, feats2)

        results["inception"] = dict(
            fid=score,
            model="InceptionV3",
            n_features = 2048
        )

    ###
    #  CLIP
    ###
    if args.clip_fid:
        logger.info("Computing FID using CLIP feature extractor")

        from cleanfid.clip_features import CLIP_fx, img_preprocess_clip
        feat_model = CLIP_fx("ViT-B/32", device=device)
        
        feats1 = cache_files(flist1, args.primary_input, "clip", feat_model, custom_fn_resize=img_preprocess_clip)
        feats2 = cache_files(flist2, args.secondary_input, "clip", feat_model, custom_fn_resize=img_preprocess_clip)

        score = fid.fid_from_feats(feats1, feats2)

        results["clip"] = dict(
            fid=score,
            model="clip_vit_b_32",
            n_features = 512
        )

    ###
    #  In-domain feature extractor (DenseNet-121)
    #  https://github.com/mlmed/torchxrayvision
    ###
    if args.xrv_fid:
        logger.info("Computing FID using in-domain feature extractor")
        try:
            import torchxrayvision as xrv

            model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device)

            class FeatureExtractor():
                def __init__(self, 
                             model: torch.nn.Module,
                             transform: Optional[Callable]):
                    self.model = model
                    if transform is not None:
                      self.transform = transform

                def __call__(self, x):
                    if self.transform is not None:
                      x = self.transform(x)
                    return self.model.features2(x)
            
            def xray_crop_center(img):
              y, x = img.shape[2:] # in BCHW format
              crop_size = torch.min(torch.tensor([y, x]))
              startx = x // 2 - (crop_size // 2)
              starty = y // 2 - (crop_size // 2)
              return img[:, starty:starty + crop_size, startx:startx + crop_size]

            class RGB2GRAY:
              def __init__(self, out_chans = 1):
                  self.out_chans = out_chans

              def __call__(self, x):
                  return transforms.functional.rgb_to_grayscale(x, self.out_chans)

            # Build torchxrayvision transform pipeline
            tf_list = []

            if True: # TODO: Implement / args.xrv_normalize:
              tf_list.extend([
                # Necessary for the pretrained XRV
                RGB2GRAY(),
                # Rescale following the original implementation:
                transforms.Lambda(lambda x: ((2048*(x/255)) - 1024)),
                # Center crop following the original implementation:
                xray_crop_center
              ])

            if True: # TODO: Implement / args.xrv_resize is not None:
              tf_list.extend([
                transforms.Resize(224)
              ])

            transform = transforms.Compose(tf_list)

            custom_extractor = FeatureExtractor(model, transform)

            feats1 = cache_files(flist1, args.primary_input, "xrv", custom_extractor)
            feats2 = cache_files(flist2, args.secondary_input, "xrv", custom_extractor)

            score = fid.fid_from_feats(feats1, feats2)

            results["xrv"] = dict(
                fid=score,
                model=str(model),
                n_features=1024,
                transform=str(transform)
            )

        except ImportError:
            logger.warning(("torchxrayvision could not be imported. " \
                        "Skipping DenseNet-121 FID calculation."))

    pprint(results)

    if output_dir is not None:
      import json, time
      ts = str(time.time()).split(".")[0]
      exp_name = args.experiment_name.lower().replace(" ","_")
      fp = Path(output_dir)/f"fid_results.{exp_name}.{ts}.json"
      with open(fp, 'w') as p:
          json.dump(results, p)
      logger.info(f"Results saved to: {fp}")

if __name__ == "__main__":
    main()
