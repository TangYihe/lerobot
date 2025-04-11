# Stephen's scripts

# bash env variables
#export HUGGINGFACE_HUB_CACHE="/svl/u/yihetang/lerobot/.hf_cache/hub"
#export HF_DATASETS_CACHE="/svl/u/yihetang/lerobot/.hf_cache/datasets"
#export LEROBOT_HOME="/svl/u/yihetang/lerobot/.hf_cache/lerobot"
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Convert R1 dataset to LERobot dataset
#python lerobot/scripts/push_r1_dataset_to_hub.py # currently manually edit the dataset name

# Training with R1 dataset
python lerobot/scripts/train.py \
--policy.path=lerobot/pi0 \
--dataset.repo_id=s-tian/galaxea_r1_cup \
--batch_size=8 \
--policy.use_3d=true \
--policy.use_2d=false