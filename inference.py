import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import argparse
import os

from tqdm import tqdm

from torch.nn import Softmax
from models.modeling import VisionTransformer, CONFIGS
from utils.model_utils import load_model_inference
from utils.dist_util import get_world_size


test_transform = transforms.Compose([
        transforms.Resize((600, 600), Image.BILINEAR),
        transforms.CenterCrop((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    if args.feature_fusion:
        config.feature_fusion=True
    config.num_token = args.num_token
    num_classes = 8
    model = VisionTransformer(
        config, args.img_size, zero_head=True, num_classes=num_classes,
        vis=True, smoothing_value=args.smoothing_value, dataset=args.dataset)

    if args.pretrained_model is not None:
        pretrained_model = torch.load(args.pretrained_model)
        model.load_state_dict(pretrained_model, strict=False)
    model.to(args.device)

    return args, model


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", default="fish", help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument("--pretrained_dir", type=str, default="ViT-L_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--pretrained_model", type=str,
                        default='./checkpoints/ViT-B_16_lr0.005_steps1640.bin',
                        help="load pretrained model")
    parser.add_argument("--img_size", default=448, type=int,
                        help="Resolution size")
    parser.add_argument("--resize_size", default=600, type=int,
                        help="Resolution size")
    parser.add_argument("--num_token", default=12, type=int,
                        help="the number of selected token in each layer, 12 for soy.loc, cotton and cub, 24 for dog.")
    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")
    parser.add_argument('--split', type=str, default='non-overlap',
                        help="Split method")
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--feature_fusion', action='store_true',
                        help="Whether to use feature fusion")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device
    args.nprocs = torch.cuda.device_count()

    # Model & Tokenizer Setup
    args, model = setup(args)
    # print(model)

    # all the testing images
    with open(os.path.join(args.data_root, 'test_order.txt')) as f:
        test_images = f.readlines()
        test_images = [line.rstrip() for line in test_images]

    f = open('submission.csv', 'w')
    f.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')

    softmax = Softmax(dim=0)

    for i, img_path in enumerate(tqdm(test_images)):
        if img_path.split('/')[0] == 'test_stg2':
            img_name = img_path
        else:
            img_name = img_path.split('/')[-1]

        img_path_full = os.path.join(args.data_root, img_path)
        img_PIL = Image.open(img_path_full).convert('RGB')
        inputs = test_transform(img_PIL)

        with torch.no_grad():
            outputs = model(
                (inputs.unsqueeze(0)).to(device))[0].squeeze(0)  # .cpu().numpy()
            # outputs = softmax(outputs).cpu().numpy()
            outputs = torch.exp(outputs).cpu().numpy()

        f.write(img_name)
        for v in outputs:
            f.write(',{}'.format(v))
        f.write('\n')

    f.close()


if __name__ == "__main__":
    main()
