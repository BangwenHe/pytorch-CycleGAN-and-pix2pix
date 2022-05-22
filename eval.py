import os
import datetime
import numpy as np
from tqdm import tqdm
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from util.metric import calculate_area, mean_iou, f1_score

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options

    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # model.eval()
    pred = []
    target = []

    for i, data in tqdm(enumerate(dataset)):
        model.set_input(data)
        model.test()

        visuals = model.get_current_visuals()  # get image results
        pred.append(visuals['fake_B'].detach().cpu().numpy())
        target.append(visuals['real_B'].detach().cpu().numpy())
    
    # print(len(pred), pred[0].shape)
    # print(len(target), target[0].shape)

    pred = np.squeeze(np.concatenate(pred, axis=0))
    target = np.squeeze(np.concatenate(target, axis=0))

    pred = pred > 0
    target = target > 0

    intersect_area, pred_area, label_area = calculate_area(pred, target, 2)
    class_iou, miou = mean_iou(intersect_area, pred_area, label_area)
    f_measure = f1_score(pred, target, 2)

    metrics = {'miou': miou, 'f_measure': f_measure}
    for i in range(2):
        metrics[f'class_{i}_iou'] = class_iou[i]

    print(metrics)

    result_dir = os.path.join(opt.results_dir, opt.name, opt.phase)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(result_dir, f'metrics_{timestamp}.txt'), 'w') as f:
        f.write(str(metrics))
