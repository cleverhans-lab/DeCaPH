# DeCaPH
We propose **De**centralized, **C**ollaborative, **a**nd **P**rivacy-preserving Machine Learning for Multi-**H**ospital Data (DeCaPH), 
a fully-decentralized and privacy-preserving collaborative
ML training framework for hospital collaborations. It provide distributed 
differential privacy (DDP) for models under the threat model for hospital collaboration (i.e., honest-but-curious).


## Desired properties:
1. it allows different parties to collaboratively train an ML model without transferring their private datasets (i.e., no data centralization);
2. it safeguards patients' privacy by limiting the potential privacy leakage arising from any contents shared across the parties during the training process;
3. it facilitates the ML model training without relying on a centralized party/server. 

### Dependency
Our code is implemented and tested on PyTorch (Python 3.6). Following packages are used:
```
numpy==1.19.5
opacus==0.15.0
pandas==1.1.5
scikit-learn
scipy==1.5.4
torch==1.10.0
torchvision==0.11.1
tqdm==4.64.0
```
We also used modified version of TorchXrayVision (version 0.0.37)

These packages can be installed with `pip install -r requirements.txt`

Or use Conda virtual-environment
```
conda create --name DeCaPH python=3.6 
conda activate DeCaPH
conda install pip
pip install -r requirements.txt
```

## Predict mortality of patients 
### GEMINI dataset 
Data cannot be made publicly available due to limitations in research ethics approvals and 
data sharing agreements, but access can be obtained upon reasonable request and in 
line with local ethics and privacy protocols, via [geminimedicine](https://www.geminimedicine.ca/).


## Cell type classification using single cell human pancreas dataset

### Download dataset
```
# create dataset dir 
mkdir dataset

# download the dataset 
curl -o dataset/Pancreas.zip https://data.wanglab.ml/OCAT/Pancreas.zip

# unzip the data
unzip dataset/Pancreas.zip -d dataset/
```

### Prepare the dataset
```
# split/prepare the dataset for 5-fold CV
python data_prep.py --dataset pancreas --dataset_path "dataset/Pancreas" --recreate_data 1 
```

### Train with DeCaPH (SVC)
``` 
num_global_server_epoch=50
target_budget=5.6
for exp_id in {0..4..1}
do
python train_DeCaPH.py --device gpu --lr 0.1 --dataset_path "dataset/Pancreas" --batch_size 128 \
--physical_batch_size 128 --architecture SVC --dataset pancreas --num_global_server_epoch ${num_global_server_epoch} \
--no_dp 0 --exp_name "svc" --log_transform 1 --weight_decay 0.0002 --max_grad_norm 0.5 \
--noise_multiplier 1.0 --exp_id ${exp_id} --seed ${exp_id} --target_budget ${target_budget}
done
```
### Train with DeCaPH (MLP)

```
num_global_server_epoch=50
target_budget=5.6
for exp_id in {0..4..1}
do
python train_DeCaPH.py --device gpu --lr 0.1 --dataset_path "dataset/Pancreas" --batch_size 128 \
--physical_batch_size 128 --architecture SVC --dataset pancreas --num_global_server_epoch ${num_global_server_epoch} \
--no_dp 0 --exp_name "mlp" --log_transform 1 --weight_decay 0.0002 --max_grad_norm 0.5 \
--noise_multiplier 1.0 --exp_id ${exp_id} --seed ${exp_id} --target_budget ${target_budget}
done


```

## Pathology identification using Chest Radiology
### download the datasets to "dataset/xray"
 * NIH 
   * (original): https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community
   * (downsized version by TorchXrayVision): https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0

 * PadChest 
   * (original): https://bimcv.cipf.es/bimcv-projects/padchest/ 
   * (downsized version by TorchXrayVision): https://academictorrents.com/details/96ebb4f92b85929eadfb16761f310a6d04105797

 * ChexPert (CheXpert-v1.0-small): https://stanfordmlgroup.github.io/competitions/chexpert/

### Prepare the dataset
```
# split/prepare the dataset for 5-fold CV
python data_prep.py --dataset xray --dataset_path "dataset/xray" --recreate_data 1
```

### Train with DeCaPH (initial state of the model pre-trained with MIMIC-CXR)
```
target_budget=0.62
num_global_server_epoch=3
for exp_id in {0..4..1}
do
python train_DeCaPH.py --device gpu --lr 0.01 --dataset_path "dataset/xray" --batch_size 64 --physical_batch_size 64 \
--architecture densenet121 --dataset xray --num_global_server_epoch ${num_global_server_epoch} --no_dp 0 \
--weight_decay 1e-5 --max_grad_norm 0.5 --noise_multiplier 1.0 --exp_id ${exp_id} \
--seed ${exp_id} --initial_model_state "mimic_ch-densenet121-0-best.pt" --num_workers 6 \
--freeze_running_stats 1 --exp_name "mimic_pretrained" --target_budget ${target_budget}
done

```

### Train with DeCaPH (initial state of the model pre-trained with ImageNet)
```
target_budget=0.65
num_global_server_epoch=7
for exp_id in {0..4..1}
do
python train_DeCaPH.py --device gpu --lr 0.01 --dataset_path "dataset/xray" --batch_size 64 --physical_batch_size 64 \
--architecture densenet121 --dataset xray --num_global_server_epoch ${num_global_server_epoch} --no_dp 0 \
--weight_decay 1e-5 --max_grad_norm 0.5 --noise_multiplier 1.0 --exp_id ${exp_id} \
--seed ${exp_id} --initial_model_state "" --num_workers 6 \
--freeze_running_stats 1 --exp_name "imgnet_pretrained" --target_budget ${target_budget}
done
```