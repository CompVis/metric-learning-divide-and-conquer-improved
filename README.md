# Improving Deep Metric Learning by Divide and Conquer
## About

PyTorch implementation for the paper _Improving Deep Metric 
Learning by Divide and Conquer_ accepted to **TPAMI** (Sep. 2021), which is our follow-up paper of
[_Divide and Conquer the Embedding Space for Metric Learning (CVPR 2019)_](https://github.com/CompVis/metric-learning-divide-and-conquer)

**Links**:
* arxiv: https://arxiv.org/abs/2109.04003 or
* TPAMI early access: https://ieeexplore.ieee.org/document/9540303


## Requirements

* PyTorch 1.1.0
* Faiss-GPU >= 1.5.0, [Link](https://github.com/facebookresearch/faiss)
* albumentations >= 0.4.5, [Link](https://github.com/albumentations-team/albumentations)


## Usage
### Training:

Training is done by using `train.py` and setting the respective flags, all of which are listed and explained 
in `/experiment/margin_loss_resnet50.py`.

**A basic sample run using default parameters would like this**:

```
python train.py --experiment margin_loss_resnet50 \
                --dataset=sop -i=$NAME -seed=4 \
                --sz-embedding=512 --mod-epoch=2 --nb-clusters=32 --nb-epochs=180 \
                --batch-size=80 --num-samples-per-class=2 --backend=faiss-gpu \
                --lr-gamma=0.3 --mod-epoch-freeze=1 --sampler=distanceweighted \
                --weight-decay=1e-4 --batch-sampler=adaptbalanced \
                --force-full-embedding-epoch=80 --mask-lr-mult=100 \
                --masking-lambda=0.01 --mask-wd-mult=1 --dataset-dir=$datadir
```

- **--dataset** specify the dataset that you want to train the model for, choose one of `--dataset=cub` (CUB200-2011),
`cars` (CARS196), `sop` (Standford Online Porducts), `inshop` (In-Shop cloths retireval) or `vid` (PKU Vehicle id).
- **--nb_clusters**: could be maximumly possible set to 16. For larger number of clusters, you may need to change the default 
limit of opened files allowed for each process. For example, if you are a ubuntu user, `ulimit -n [twice the number of the default]` usually will do the trick. 
Besides, also consider the total number of different classes in your dataset and the sampling strategy when you setting this value.
- **--experiment margin_loss_resnet50** please keep this untouched, otherwise the args won't be read correctly.
- **--dataset-dir**: the path to the datasets, check the *Datasets* section below.
- **--sampler**: batchminer used to sample pairs or triplets (in embedding space) to create learning signal, check `/metriclearning/sampler` for details.
- **--batch-sampler**: data sampler used to generate training batches, check `/dataset/sampler.py` for details.
- **--nb-epochs**: the maximum training epochs.
- **--mod-epoch**: division frequency in the paper - the number of training epochs between consecutive divisions.

_Note:_ For exact settings for different datasets, please check the original paper. For arguments not mentioned above, 
please check `/experiment/margin_loss_resnet50.py` for explanation.


### Evaluate or check results during:

* **evaluate a trained model**: `eval_model.py <log and model checkpoint path>`. It is suggested to put only those models (checkpoints and logs) you want to eval into one folder otherwise it will evaluate all the models ind the folder

* **check intermediate results**: the model checkpoints and log files are saved in the selected log-directory (by default: `/log`). 
You can print a summary of the results with `python browse_results <log path>`.
  

### Datasets:

The method is tested on the following datasets:

* CUB200-2011 (http://www.vision.caltech.edu/visipedia/CUB-200.html)
* CARS196 (https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
* Stanford Online Products (http://cvgl.stanford.edu/projects/lifted_struct/)
* In-shop Clothes Retrieval Benchmark (http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
* PKU VehicleID (https://www.pkuml.org/resources/pku-vehicleid.html)

Assuming your folder is placed in e.g. `<$datadir/cars196>`, pass `$datadir` as input to `--dataset-dir`, by default. To avoid conflicts between the folder structure and our pipeline, please make sure that the datasets 
have the following internal structure:

* For CUB200-2011,
```
cub-200-2011
└───images
|    └───001.Black_footed_Albatross
|           │   Black_Footed_Albatross_0001_796111
|           │   ...
|    ...
```
* For Cars196, please download the tar of all images, all bounding boxes, labels for both training and test 
[here](http://ai.stanford.edu/~jkrause/cars/car_dataset.html) and unzip them, which is consistent with our dataloader.
```
cars196
└───car_ims
|    │016180.jpg
|    │   ...
|    │016185.jpg
└───cars_train
|    │08092.jpg
|    │   ...
|    │08144.jpg
└───cars_annos.mat

```

* For Stanford Online Products:
```
sop
└───bicycle_final
|   │   111085122871_0.jpg
|           ...
|...
└───cabinet_final
|   │   xxx.jpg
|       ...
| bicycle.txt
| ...
| cabinet_final.txt
```

* For In-shop Clothes and PKU Vehicle id datasets, simply download them from the above given links and unzip their folder as they originally are.


## Results

__CUB200__

Variants | Loss/Sampling	|   NMI  |  mARP  | Recall @ 1 -- 2 -- 4 -- 8
---------|---------------	|--------|------|-----------------
fixed |  Margin/Distance    | 68.60 | 54.98 | 67.39 -- 77.43 --  84.82 --  90.83    
learn |  Margin/Distance   	| 69.84 | 55.11 | __67.71__ --  78.14 --  85.80 --  91.46


__Cars196__

Variants | Loss/Sampling	|   NMI  |  mARP  | Recall @ 1 -- 2 -- 4 -- 8
---------|---------------	|--------|------|-----------------
fixed |  Margin/Distance    | 70.57 | 65.89 | __87.22__ -- 92.19 --  95.39 --  97.43    
learn |  Margin/Distance   	| 70.10 | 65.54 | 86.90 --  92.34 --  95.41 --  97.45


__In-Shop Clothes__

Variants | Loss/Sampling    |   NMI  |  mARP  | Recall @ 1 -- 10 -- 20 -- 30
------|---------------      |--------|------|-----------------
fixed |  Margin/Distance	| 89.76 | 87.86| 89.84 -- 97.56 -- 98.21 -- 98.51
learn |  Margin/Distance    | 89.88 | 88.50| __90.49__ -- 97.48 -- 98.23 -- 98.51

__Online Products__

Variants | Loss/Sampling	|   NMI  |  mARP  | Recall @ 1 -- 10 -- 100
---------|---------------	|--------|------|-----------------
fixed |  Margin/Distance    | 89.59 | 79.32| 79.50 -- 90.44 -- 95.08    
learn |  Margin/Distance   	| 89.68 | 79.60| __79.80__ --  90.46 --  95.26


__VID (Large eval set)__ 

Variants | Loss/Sampling	|   NMI  |  mARP  | Recall @ 1 -- 5
------|---------------      |--------|------|-----------------
fixed |  Margin/Distance	| 90.78 | 92.89 | __94.01__ -- 96.18
learn |  Margin/Distance    | 90.70 | 92.68 | 93.93 -- 95.99 


_Disclaimer: The results above are slightly different from those on the paper, as the results in the paper were obtained before code refactoring and 
with PyTorch (0.4.1) and Faiss (< 1.5). So there may be small deviations in results based on the Software (different PyTorch/Cuda versions) and Hardware (e.g. between P100 and RTX GPUs) used to run these experiments._


## Related Repos

* Divide and Conquer the Embedding Space for Metric Learning (our previous paper on CVPR 2019): 
https://github.com/CompVis/metric-learning-divide-and-conquer
* An easy-to-use repo to start your DML research, containing collections of models, losses, and samplers implemented in PyTorch:
https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch

## BibTex
If you use this code in your research, please cite the following papers:

```
@InProceedings{dcesml,
  title={Divide and Conquer the Embedding Space for Metric Learning},
  author={Sanakoyeu, Artsiom and Tschernezki, Vadim and B\"uchler, Uta and Ommer, Bj\"orn},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}

@article{sanakoyeu2021improving,
  title={Improving Deep Metric Learning by Divide and Conquer}, 
  author={Artsiom Sanakoyeu and Pingchuan Ma and Vadim Tschernezki and Björn Ommer},
  journal={IEEE Transactions on pattern analysis and machine intelligence},
  year={2021},
  publisher={IEEE}
}
```
