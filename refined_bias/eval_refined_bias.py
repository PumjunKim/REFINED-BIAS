import os
import json
import argparse

import torch
import numpy as np
from tqdm import tqdm

from utils import get_model_list ,load_dataset, load_model, set_class_mappings, top1_acc, mrr_acc

script_dir = os.path.dirname(os.path.abspath(__file__))


def main(args):
    # Define GPU device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define decision mapping
    id_to_class, class_mapping = set_class_mappings(args.dataset)

    # Get model names to eval
    model_names = get_model_list(target=args.across)

    # Load dataset and model
    data_loader = load_dataset(dataset=args.dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    print(f"===================================== INFERENCE =====================================")
    # GT label
    gt_label =[]

    # Store class prediction ranks for each model (sorted by descending probabilities)
    model_prediction_ranks = {}

    # Inference imagenet-pretrained model
    pbar = tqdm(model_names, dynamic_ncols=True, ncols=0)
    for model_name in pbar:
        pbar.set_description(f"Inference {model_name}")

        package = 'torchvision' if args.across == 'arch' else 'timm'
        model = load_model(model_name, package, device=device)

        model_prediction_ranks[model_name] = []
        for images, targets, paths in data_loader:
            with torch.no_grad():
                images = images.to(device)
                logits = model(images)
                softmax_output = torch.nn.functional.softmax(logits, dim=1)
                softmax_np = softmax_output.cpu().numpy()
                if model_name == model_names[0]:
                    gt_label.extend(targets)

            # Get and store ranks for predictions
            model_prediction_ranks[model_name].extend(class_mapping(softmax_np))
    
    model_prediction_ranks = {model_name: [[id_to_class.get(itm, itm) for itm in arr] for arr in model_prediction_ranks[model_name]] for model_name in model_names}

    print(f"===================================== RESULTS =====================================")
    # Get bias score based on MRR
    bias_score = mrr_acc(model_prediction_ranks, gt_label)

    print(f"REFINED-BIAS {args.dataset.split('_')[2].capitalize()} Bias on {args.across}")
    max_name_len = max(len(name) for name in model_names)
    for model_name in model_names:
        score = bias_score[model_name]['score']
        print(f"> {model_name:<{max_name_len}} : {score:.4f}")
    
    dir_name = 'across_model_architecture' if args.across == 'arch' else 'across_learning_strategy'
    result_dir = f"{os.path.join(script_dir, 'results', dir_name)}"
    
    os.makedirs(result_dir, exist_ok=True)
    with open(f"{result_dir}/{args.dataset}_mrr.json", "w") as f:
        json.dump(bias_score, f, indent=4)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)  # RB-edge, RB-texture, cue-conflict
    parser.add_argument('--across', type=str)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    # parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    main(args)