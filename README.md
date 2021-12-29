# FFVT

## Installation

* The following packages are required to run the scripts:
  - [Python >= 3.6]
  - [PyTorch = 1.5]
  - [Torchvision]

* Install [Apex](https://github.com/NVIDIA/apex)
* Install other needed packages
```
pip install -r requirements.txt
```

## Prepare data

* Put `train`, `test_stg1` and `test_stg2` under `data`
* run  `prepare_data.py`
```
cd data
python prepare_data.py
```
* Three txt files will be generated:
1. train_image_labels.txt
2. valid_image_labels.txt
3. test_order.txt

## Download Google pre-trained ViT models

* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): ViT-B_16, ViT-B_32...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz
```

## Train
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node 4 train.py --name {exp_name} --model_type ViT-B_16 --train_batch_size 8 --learning_rate 0.0005 --fp16 --eval_every 20 --feature_fusion
```

## Inference
* Get our trained model [here](https://drive.google.com/file/d/104uZv9ZKWDhNuwHupQobG9UtCb8LeX0M/view?usp=sharing)
* Put it under `output/ViT-B_16_lr0.005/`
* More information in `output/ViT-B_16_lr0.005/README.md`
```
CUDA_VISIBLE_DEVICES=0 python inference.py --name test
```

## Submission to Kaggle competition
```
kaggle competitions submit -c the-nature-conservancy-fisheries-monitoring -f submission.csv -m "{submission description}"
```

