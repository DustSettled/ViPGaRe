#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = True
        self.data_device = "cuda"
        self.eval = True
        self.load2gpu_on_the_fly = True
        self.is_blender = True
        self.is_6dof = False
        self.max_time = 0.7
        self.light = False
        self.physics_code = 16
        self.use_affine = True  # 默认使用12-DOF Affine速度场
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 40_000
        self.warm_up = 3_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.deform_lr_max_steps = 40_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.001
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.pinn_weight = -1.
        self.kmeans_weight = -1.
        self.gradual = True
        self.use_mpm = True
        self.mpm_weight = 0.1
        self.mpm_elastic_weight = 0.01
        self.mpm_grid_res = 256
        super().__init__(parser, "Optimization Parameters")


class PINNsParams(ParamGroup):
    """
    Physics-Informed Neural Networks (PINNs) Parameters

    Controls physics constraints for velocity field regularization.
    """
    def __init__(self, parser):
        # PINNs training schedule
        self.pinns_start_iter = 3000  # Start PINNs training at iteration 3000
        self.pinns_curriculum_end_iter = 40000 # Reach full weight at iteration 30000
        self.pinns_max_weight = 0.1  # Maximum weight for PINNs loss

        # Scene type and equation selection
        self.pinns_mode = "mixed"  # Options: 'rigid', 'fluid', 'mixed'

        # Equation weights (manual tuning)
        self.pinns_weight_transport = 1.0  # Transport equation weight
        self.pinns_weight_divergence = 0.1  # Divergence-free constraint weight
        self.pinns_weight_rigid = 0.5  # Rigid body constraint weight
        self.pinns_weight_energy = 0.1  # Energy conservation weight

        # Sampling parameters - 关键修复：大幅降低采样率避免509GB显存爆炸
        self.pinns_sample_ratio = 0.05  # Base sampling ratio for t ≤ 0.75 (从0.05→0.01)
        self.pinns_extrap_sample_ratio = 0.1  # Sampling ratio for t > 0.75 (从0.1→0.02)
        self.pinns_extrap_threshold = 0.75  # Time threshold for extrapolation
        self.pinns_knn_update_interval = 10  # Update KNN cache every N iterations
        self.pinns_perturbation_std = 0.05  # Spatial perturbation std

        # Physics correction at inference
        self.pinns_physics_correct = False  # Apply physics correction during inference

        super().__init__(parser, "PINNs Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
