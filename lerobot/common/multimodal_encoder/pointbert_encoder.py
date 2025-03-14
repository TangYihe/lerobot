import torch
import torch.nn as nn
import os
import sys
import yaml
from easydict import EasyDict
import logging
from contextlib import nullcontext
from lerobot.common.multimodal_encoder.pointbert.point_encoder import PointTransformer

def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config

def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        new_config = yaml.load(f, Loader=yaml.FullLoader)
    merge_new_config(config=config, new_config=new_config)
    return config

class PointBERT(nn.Module):
    def __init__(self, use_color=True, ckpt_path=None, fix_pointnet=True):
        super().__init__()
        
        if ckpt_path is None:
            ckpt_path = "/svl/u/zeyanjie/g1_vla/RoboticsDiffusionTransformer/third_party/point_bert_v1.2.pt"
            # ckpt_path = "third_party/point_bert_v1.2.pt"
        
        self.ckpt_path = ckpt_path
        self.use_color = use_color
        self.fix_pointnet = fix_pointnet

        # * default for v1.2, v1.1 uses PointTransformer_base_8192point.yaml
        config_v11 = "PointTransformer_base_8192point.yaml"
        config_v12 = "PointTransformer_8192point_2layer.yaml"
        
        # address of config file, in the same dir of this file
        point_bert_config_name = config_v12 
        point_bert_config_addr = os.path.join(os.path.dirname(__file__), "pointbert", point_bert_config_name)
        logging.info(f"[PointBERT] Initializing PointBERT config from {point_bert_config_addr}.", "green")
        point_bert_config = cfg_from_yaml_file(point_bert_config_addr)
        if self.use_color:
            point_bert_config.model.point_dims = 6
        use_max_pool = getattr(point_bert_config.model, "use_max_pool", False) # * default is false
        
        self.point_backbone = PointTransformer(point_bert_config.model, use_max_pool=use_max_pool)
        
        self.point_backbone_config = {
                "point_cloud_dim": point_bert_config.model.point_dims,
                "backbone_output_dim": point_bert_config.model.trans_dim if not use_max_pool else point_bert_config.model.trans_dim * 2,
                "point_token_len": point_bert_config.model.num_group + 1 if not use_max_pool else 1, # * number of output features, with cls token
                "projection_hidden_layer": point_bert_config.model.get('projection_hidden_layer', 0),
                "use_max_pool": use_max_pool
            }
        
        if point_bert_config.model.get('projection_hidden_layer', 0) > 0:
            self.point_backbone_config["projection_hidden_dim"] = point_bert_config.model.projection_hidden_dim # a list
        
        logging.info(f"[PointBERT] PointBERT config: {self.point_backbone_config}", "green")
         
        self.load_point_backbone_checkpoint(self.ckpt_path)
        logging.info(f"[PointBERT] Loading PointBERT checkpoint from {self.ckpt_path}.", "green")
        
        if self.fix_pointnet:
            self.point_backbone.eval()
    
    
    def load_point_backbone_checkpoint(self, checkpoint_path):
        self.point_backbone.load_checkpoint(checkpoint_path)
    
    def forward(self, point_cloud):
        with torch.no_grad() if self.fix_pointnet else nullcontext():
            # logging.info(f"[pointbert_encoder] sampled point_cloud: {point_cloud.shape}", "green")  
            if self.fix_pointnet:
                self.point_backbone.eval()
            if len(point_cloud.shape) == 3:
                point_features = self.point_backbone(point_cloud)
            elif len(point_cloud.shape) == 4:
                B, T, N, C = point_cloud.shape
                point_features = self.point_backbone(point_cloud.reshape(B*T, N, C))
                _, N, D = point_features.shape
                point_features = point_features.reshape(B, T*N, D)
            else:
                raise ValueError(f"Invalid point cloud shape: {point_cloud.shape}")
            # logging.info(f"[pointbert_encoder] sampled point_features: {point_features.shape}", "green")
        return point_features
