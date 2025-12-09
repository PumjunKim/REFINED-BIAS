import warnings
from abc import ABC
from glob import glob

import numpy
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.model_zoo import load_pretrained_model_timm, load_pretrained_model_torchvision
from decision_mapping.decision_mappings import ImageNetProbabilitiesTo1000ClassesMapping

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torchvision.models._utils"
)

####################################################################
#   Get ILSVRC 2012 wnIDs                 
####################################################################
def set_class_mappings(dataset):
    class_mapping = ImageNetProbabilitiesTo1000ClassesMapping()
    if 'shape' in dataset:
        id_to_class = {"n02978881": "cassette", "n03584254": "ipod", 
                        "n02782093": "balloon", "n04086273": "revolver",
                        "n03196217": "clock", "n02708093": "clock", 
                        "n04548280": "clock","n03793489": "mouse", 
                        "n03976467": "camera", "n04069434": "camera",
                        "n06596364": "book", "n04254680": "soccer_ball", 
                        "n03544143": "hourglass",}
    elif 'texture' in dataset:
        id_to_class = { "n01917289": "brain_coral",
                        "n03207743": "dishrag", "n02129604": "tiger",
                        "n02346627": "porcupine", "n02391049": "zebra",
                        "n07714990": "broccoli", "n02130308": "cheetah",
                        "n02504458": "elephant", "n02504013": "elephant",
                        "n07745940": "strawberry", "n03530642": "honeycomb"}

    return id_to_class, class_mapping
####################################################################


####################################################################
#   Model Name Loader                     
####################################################################
def get_model_list(target):
    if target == 'arch':
        model_names = ['bagnet9', 'bagnet17', 'bagnet33',
                    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", 
                    "alexnet", "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn",
                    "densenet121", "densenet169", "densenet201",
                    "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
                    "vit_small_patch16_224", "vit_base_patch16_224", "vit_large_patch16_224",]
    elif target == 'strategy':
        model_names = ["tv_resnet50", "tv2_resnet50", "resnet50.d_in1k", "resnet50.c1_in1k", "resnet50.b1k_in1k", "resnet50.c2_in1k", "resnet50.b2k_in1k", "resnet50.a3_in1k", "resnet50.a2_in1k", 
                    "resnet50.a1h_in1k", "resnet50.a1_in1k", "resnet50_diffusionnoise_fixed_nonoise", "resnet50_prime", "resnet50_deepaugment", "resnet50_noisymix", "resnet50_deepaugment_augmix", 
                    "resnet50_pixmix_90ep", "resnet50_pixmix_180ep", "resnet50_augmix_180ep", "resnet50_tsbias_tbias", "resnet50_tsbias_sbias", "resnet50_tsbias_debiased", "resnet50_opticsaugment", 
                    "resnet50_trained_on_SIN", "resnet50_trained_on_SIN_and_finetuned_on_IN", "resnet50_trained_on_SIN_and_IN", "resnet50_frozen_random", "robust_resnet50_l2_eps0.01", 
                    "robust_resnet50_l2_eps0.03", "robust_resnet50_l2_eps0.05", "robust_resnet50_l2_eps0.1", "robust_resnet50_l2_eps0.25", "robust_resnet50_l2_eps0.5", "robust_resnet50_l2_eps1", 
                    "robust_resnet50_linf_eps8.0", "robust_resnet50_l2_eps5", "robust_resnet50_linf_eps0.5", "robust_resnet50_linf_eps1.0", "robust_resnet50_linf_eps2.0", "robust_resnet50_linf_eps4.0", 
                    "robust_resnet50_l2_eps0", "robust_resnet50_l2_eps3", "resnet50_moco_v3_100ep", "resnet50_moco_v3_300ep", "resnet50_moco_v3_1000ep", "resnet50_dino", "resnet50_swav", "resnet50_simclrv2"]
    else:
        print("")
    return model_names
####################################################################


####################################################################
#   Image Path Mapping & DataLoader       
####################################################################
class ImagePathToInformationMapping(ABC):
    def __init__(self):
        pass

    def __call__(self, full_path):
        pass


class ImageNetInfoMapping(ImagePathToInformationMapping):
    """
        For ImageNet-like directory structures without sessions/conditions:
        .../{category}/{img_name}
    """

    def __call__(self, full_path):
        session_name = "session-1"
        img_name = full_path.split("/")[-1]
        condition = "NaN"
        category = full_path.split("/")[-2]

        return session_name, img_name, condition, category


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder

    Adapted from:
    https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """

    def __init__(self, *args, **kwargs):

        if "info_mapping" in kwargs.keys():
            self.info_mapping = kwargs["info_mapping"]
            del kwargs["info_mapping"]
        else:
            self.info_mapping = ImageNetInfoMapping()

        super(ImageFolderWithPaths, self).__init__(*args, **kwargs)


    def __getitem__(self, index):
        (sample, target) = super(ImageFolderWithPaths, self).__getitem__(index)
        
        path = self.imgs[index][0]
        _, _, _, new_target = self.info_mapping(path)
        original_tuple = (sample, new_target)

        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


#& dataloader function
def load_dataset(dataset, *args, **kwargs):  
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    dataset_path = f'./datasets/{dataset}/'
    dataset_obj = ImageFolderWithPaths(root=dataset_path, transform=transform)

    print("=====================================  DATASET  =====================================")
    print(f"Path: {dataset_path}")
    print(f"✔ Total images: {len(dataset_obj):,}")  

    # Define DataLoader
    loader = DataLoader(
        dataset_obj,
        batch_size=kwargs.get("batch_size", 32),
        shuffle=False,
        num_workers=kwargs.get("num_workers", 4),
        pin_memory=True
    )
    return loader
####################################################################


####################################################################
#   Get model list for inference          
####################################################################
def load_model(model_name, package, pretrained=True, device=None):
    if package == 'timm':
        model = load_pretrained_model_timm(model_name)
    elif package == 'torchvision':
        model = load_pretrained_model_torchvision(model_name)
    else:
        print("Check the package..")
    model.to(device)
    model.eval()
    return model
####################################################################


####################################################################
#   Evaluation metric for top-1 & mrr     
####################################################################
def top1_acc(rank, gt_label):
    classes = list(set(gt_label))
    model_score = {}
    for model in tqdm(rank.keys()):
        model_prediction = rank[model]
        total_data_len = len(model_prediction)
        num_cor = 0
        class_match = {}
        for CLS in classes:
            class_match[CLS] = 0
        for idx in range(total_data_len):
            top_prediction = model_prediction[idx][0]
            if (top_prediction == gt_label[idx]):
                num_cor += 1
                class_match[gt_label[idx]] += 1
        model_score[model] = {"score": num_cor/total_data_len, "class_match": class_match}
    return model_score


#& rank-based evaluation metric
def mrr_acc(rank, gt_label):
    classes = list(set(gt_label))
    model_score = {}
    for model in tqdm(rank.keys()):
        model_prediction = rank[model]
        total_mrr = []
        classwise_mrr = {CLS: [] for CLS in classes}

        for idx in range(len(model_prediction)):
            predicted_rank = np.where(np.array(model_prediction[idx]) == gt_label[idx])[0][0] + 1
            total_mrr.append(1/predicted_rank)
            classwise_mrr[gt_label[idx]].append(1/predicted_rank)
            
        for CLS in classes:
            classwise_mrr[CLS] = np.mean(classwise_mrr[CLS])
        model_score[model] = {"score": np.mean(total_mrr), "class_match": classwise_mrr}
    return model_score
####################################################################