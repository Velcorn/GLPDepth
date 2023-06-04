'''
Doyeon Kim, 2022
Jan Willruth, 2023
'''

import os
from collections import OrderedDict
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import utils.logging as logging
from configs.test_options import TestOptions
from models.model import GLPDepth
from glob import glob

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']


def main():
    # experiments setting
    opt = TestOptions()
    args = opt.initialize().parse_args()
    print(args)

    if args.gpu_or_cpu == 'gpu':
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    if args.save_eval_pngs or args.save_visualize:
        result_path = './results'
        logging.check_and_make_dirs(result_path)
        print("Saving result images in to %s" % result_path)

    if args.do_evaluate:
        result_metrics = {}
        for metric in metric_name:
            result_metrics[metric] = 0.0

    print("\n1. Define Model")
    model = GLPDepth(max_depth=args.max_depth, is_train=False).to(device)
    model_weight = torch.load(args.ckpt_dir)
    if 'module' in next(iter(model_weight.items()))[0]:
        model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
    model.load_state_dict(model_weight)
    model.eval()

    print("\n2. Inference & Evaluate")
    # JW: Implement a basic iteration over a nested folder structure to load images
    img_path = '../BA/data-temp/chalearn/249-40/image'
    img_names = glob(f'{img_path}/*/001/M_00001/*.jpg')
    print(f'Found {len(img_names)} images')
    for i, img_name in enumerate(img_names):
        transform = transforms.ToTensor()
        img = cv2.imread(img_name)
        img = cv2.resize(img, (320, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_RGB = transform(img).to(device)
        # Add a 1 dimension to the tensor
        input_RGB = input_RGB.unsqueeze(0)

        with torch.no_grad():
            pred = model(input_RGB)
        pred_d = pred['pred_d']

        # Create base folder if it doesn't exist
        output_path = f'{result_path}/{"/".join(img_name.split("/")[6:-1])}'
        os.makedirs(output_path, exist_ok=True)
        filename = os.path.basename(img_name).split('.')[0]
        save_path = f'{output_path}/{filename}.png'
        pred_d_numpy = pred_d.squeeze().cpu().numpy()
        pred_d_numpy = (pred_d_numpy / pred_d_numpy.max()) * 255
        # JW: Remove color map and resize to original image size
        pred_d_numpy = pred_d_numpy.astype(np.uint8)
        pred_d_numpy = cv2.resize(pred_d_numpy, (320, 240))
        # pred_d_color = cv2.applyColorMap(pred_d_numpy, cv2.COLORMAP_RAINBOW)
        cv2.imwrite(save_path, pred_d_numpy)
        logging.progress_bar(i, len(img_names), 1, 1)

    print("Done")


if __name__ == "__main__":
    main()
