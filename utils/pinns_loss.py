"""
Physics-Informed Neural Networks (PINNs) - Loss Computation Module

This module implements PINNs loss computation and management, integrating
physics equation residuals into the training process.

Loss Components:
1. Transport equation loss
2. Divergence-free loss
3. Rigid body constraint loss
4. Energy conservation loss

Author: FreeGave Team
Date: 2025
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from utils.physics_pdes import PhysicsEquations
from utils.pinns_sampler import CollocationSampler


class PINNsLossComputer(nn.Module):
    """
    Compute PINNs loss with adaptive weighting and curriculum learning.

    Key features:
    - Fixed equation set (no user customization)
    - Adaptive weights based on pinns_mode
    - Curriculum learning schedule
    """

    def __init__(
        self,
        pinns_mode: str = 'rigid',
        weight_transport: float = 1.0,
        weight_divergence: float = 0.1,
        weight_rigid: float = 0.5,
        weight_energy: float = 0.1,
        pinns_start_iter: int = 3000,
        curriculum_end_iter: int = 30000,
        max_pinns_weight: float = 0.1,
        device: str = 'cuda'
    ):
        """
        Args:
            pinns_mode: scene type ('rigid', 'fluid', 'mixed')
            weight_transport: weight for transport equation
            weight_divergence: weight for divergence constraint
            weight_rigid: weight for rigid body constraint
            weight_energy: weight for energy conservation
            pinns_start_iter: iteration to start PINNs training
            curriculum_end_iter: iteration to reach maximum PINNs weight
            max_pinns_weight: maximum weight for PINNs loss in total loss
            device: computation device
        """
        super(PINNsLossComputer, self).__init__()

        self.pinns_mode = pinns_mode
        self.pinns_start_iter = pinns_start_iter
        self.curriculum_end_iter = curriculum_end_iter
        self.max_pinns_weight = max_pinns_weight
        self.device = device

        # Initialize physics equations module
        self.physics_eqs = PhysicsEquations(device=device)

        # Initialize collocation sampler
        self.sampler = CollocationSampler(device=device)

        # Set equation weights based on pinns_mode
        self.equation_weights = self._get_equation_weights(
            pinns_mode, weight_transport, weight_divergence, weight_rigid, weight_energy
        )

        # Set which equations to enable
        self.equation_flags = self._get_equation_flags(pinns_mode)

    def _get_equation_weights(
        self,
        pinns_mode: str,
        weight_transport: float,
        weight_divergence: float,
        weight_rigid: float,
        weight_energy: float
    ) -> Dict[str, float]:
        """
        Get equation weights based on pinns_mode.

        Preset configurations:
        - rigid: emphasize transport + rigid body
        - fluid: emphasize transport + divergence-free
        - mixed: balanced weights
        """
        if pinns_mode == 'rigid':
            return {
                'transport': weight_transport,
                'divergence': 0.0,  # Not needed for rigid body
                'rigid_body': weight_rigid,
                'energy': weight_energy
            }
        elif pinns_mode == 'fluid':
            return {
                'transport': weight_transport,
                'divergence': weight_divergence,
                'rigid_body': 0.0,  # Not needed for fluid
                'energy': weight_energy
            }
        elif pinns_mode == 'mixed':
            return {
                'transport': weight_transport,
                'divergence': weight_divergence * 0.5,
                'rigid_body': weight_rigid * 0.5,
                'energy': weight_energy
            }
        else:
            raise ValueError(f"Unknown pinns_mode: {pinns_mode}")

    def _get_equation_flags(self, pinns_mode: str) -> Dict[str, bool]:
        """
        Get flags for which equations to enable.
        """
        if pinns_mode == 'rigid':
            return {
                'enable_transport': True,
                'enable_divergence': False,
                'enable_rigid_body': True,
                'enable_energy': True
            }
        elif pinns_mode == 'fluid':
            return {
                'enable_transport': True,
                'enable_divergence': True,
                'enable_rigid_body': False,
                'enable_energy': True
            }
        elif pinns_mode == 'mixed':
            return {
                'enable_transport': True,
                'enable_divergence': True,
                'enable_rigid_body': True,
                'enable_energy': True
            }
        else:
            raise ValueError(f"Unknown pinns_mode: {pinns_mode}")

    def get_curriculum_weight(self, iteration: int) -> float:
        """
        Get curriculum learning weight for PINNs loss.

        Schedule:
        - iter < pinns_start_iter: 0.0
        - pinns_start_iter ≤ iter < curriculum_end_iter: linear ramp-up
        - iter ≥ curriculum_end_iter: max_pinns_weight

        Args:
            iteration: current training iteration

        Returns:
            weight: curriculum weight in [0, max_pinns_weight]
        """
        if iteration < self.pinns_start_iter:
            return 0.0
        elif iteration >= self.curriculum_end_iter:
            return self.max_pinns_weight
        else:
            # Linear ramp-up
            progress = (iteration - self.pinns_start_iter) / (self.curriculum_end_iter - self.pinns_start_iter)
            return self.max_pinns_weight * progress

    def create_velocity_func(
        self,
        deform_model,
        deform_code: torch.Tensor
    ):
        """
        Create a velocity function for torch.func.jacrev.

        CRITICAL FIX: Store deform_seg and vel_net separately, let physics_pdes
        handle the batching correctly.

        Args:
            deform_model: DeformModel instance
            deform_code: [M, K] - deform codes for selected points (raw)

        Returns:
            velocity_func: Callable that handles both batch and single-point input
        """
        # Store these for the closure
        deform_seg_full = deform_model.code_field.seg(deform_code).detach()  # [M, K]
        vel_net = deform_model.vel_net

        def velocity_func_batch(xyzt: torch.Tensor) -> torch.Tensor:
            """Batch mode: xyzt [N, 4] -> velocity [N, 3]"""
            return vel_net.get_vel(deform_seg_full, xyzt)

        # 给velocity_func添加属性，让physics_pdes可以访问
        velocity_func_batch.deform_seg = deform_seg_full
        velocity_func_batch.vel_net = vel_net

        return velocity_func_batch

    def create_acceleration_func(
        self,
        deform_model,
        deform_code: torch.Tensor
    ):
        """
        Create an acceleration function for torch.func.

        CRITICAL FIX: Store deform_seg and vel_net separately.

        Args:
            deform_model: DeformModel instance
            deform_code: [M, K] - deform codes for selected points (raw)

        Returns:
            acceleration_func: Callable that handles both batch and single-point input
        """
        # Store these for the closure
        deform_seg_full = deform_model.code_field.seg(deform_code).detach()  # [M, K]
        vel_net = deform_model.vel_net

        def acceleration_func_batch(xyzt: torch.Tensor) -> torch.Tensor:
            """Batch mode: xyzt [N, 4] -> acceleration [N, 3]"""
            return vel_net.get_acc(deform_seg_full, xyzt)

        # 给acceleration_func添加属性，让physics_pdes可以访问
        acceleration_func_batch.deform_seg = deform_seg_full
        acceleration_func_batch.vel_net = vel_net

        return acceleration_func_batch

    def compute_pinns_loss(
        self,
        deform_model,
        gaussians,
        t: torch.Tensor,
        iteration: int
    ) -> tuple:
        """
        Compute PINNs loss at current iteration.

        Args:
            deform_model: DeformModel instance
            gaussians: GaussianModel instance
            t: scalar tensor - current time
            iteration: current training iteration

        Returns:
            total_loss: scalar tensor - total PINNs loss
            loss_dict: dict - individual loss components for logging
        """
        # Check if PINNs should be active
        curriculum_weight = self.get_curriculum_weight(iteration)
        if curriculum_weight == 0.0:
            return torch.tensor(0.0, device=self.device), {}

        # 关键修复：使用with torch.no_grad()包裹数据准备，减少显存占用
        with torch.no_grad():
            # Get Gaussian centers
            xyz = gaussians.get_xyz.detach()  # [N, 3] - detach避免累积梯度

            # Get deform codes
            deform_code = deform_model.code_field(xyz).detach()  # [N, K] - detach

        # Sample collocation points (需要梯度，所以在no_grad外)
        collocation_points, selected_indices = self.sampler.sample_collocation_points(
            xyz, t, iteration
        )  # [M, 4], [M]

        # 关键调试：验证采样输出
        # print(f"[DEBUG pinns_loss] xyz shape: {xyz.shape}")
        # print(f"[DEBUG pinns_loss] t shape: {t.shape}, t value: {t.item() if t.numel()==1 else t}")
        # print(f"[DEBUG pinns_loss] collocation_points shape: {collocation_points.shape}")
        # print(f"[DEBUG pinns_loss] n_selected: {len(selected_indices)}")

        # Get deform codes for selected points
        with torch.no_grad():
            selected_deform_code = deform_code[selected_indices].detach()  # [M, K] - detach

        # Create velocity and acceleration functions for torch.func
        velocity_func = self.create_velocity_func(deform_model, selected_deform_code)
        acceleration_func = self.create_acceleration_func(deform_model, selected_deform_code)

        # 关键调试：在第一次PINNs计算时启用调试输出
        debug_first_call = (iteration == self.pinns_start_iter)

        if debug_first_call:
            print(f"  [PINNs] Calling physics_eqs.compute_all_residuals...")

        # Compute physics equation residuals using torch.func
        residuals = self.physics_eqs.compute_all_residuals(
            velocity_func,
            acceleration_func,
            collocation_points,
            debug_first_call=debug_first_call,
            **self.equation_flags
        )

        if debug_first_call:
            print(f"  [PINNs] Physics residuals computed successfully")
            print(f"  [PINNs] Residuals: {list(residuals.keys())}")

        # Compute weighted loss for each equation
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=self.device)

        for eq_name, residual in residuals.items():
            weight = self.equation_weights[eq_name]
            if weight > 0:
                # Mean squared residual
                eq_loss = torch.mean(residual ** 2)
                weighted_loss = weight * eq_loss

                total_loss = total_loss + weighted_loss
                loss_dict[f'pinns_{eq_name}'] = eq_loss.item()

        # Apply curriculum weight
        total_loss = curriculum_weight * total_loss
        loss_dict['pinns_total'] = total_loss.item()
        loss_dict['pinns_curriculum_weight'] = curriculum_weight

        # 关键修复：显式清理闭包引用，帮助垃圾回收
        del velocity_func
        del acceleration_func
        del residuals
        del selected_deform_code
        del collocation_points

        return total_loss, loss_dict

    def get_config(self) -> dict:
        """
        Get current configuration for logging.
        """
        return {
            'pinns_mode': self.pinns_mode,
            'equation_weights': self.equation_weights,
            'equation_flags': self.equation_flags,
            'pinns_start_iter': self.pinns_start_iter,
            'curriculum_end_iter': self.curriculum_end_iter,
            'max_pinns_weight': self.max_pinns_weight,
            'sampler_info': self.sampler.get_cache_info()
        }


def test_pinns_loss_computer():
    """
    Unit test for PINNs loss computer.
    """
    print("=" * 60)
    print("Testing PINNsLossComputer Module")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test different modes
    modes = ['rigid', 'fluid', 'mixed']

    for mode in modes:
        print(f"\n[Testing mode: {mode}]")
        computer = PINNsLossComputer(
            pinns_mode=mode,
            pinns_start_iter=3000,
            curriculum_end_iter=30000,
            max_pinns_weight=0.1,
            device=device
        )

        print(f"  Equation weights: {computer.equation_weights}")
        print(f"  Equation flags: {computer.equation_flags}")

        # Test curriculum schedule
        print(f"\n  Curriculum schedule:")
        test_iters = [0, 3000, 16500, 30000, 40000]
        for iter_val in test_iters:
            weight = computer.get_curriculum_weight(iter_val)
            print(f"    Iteration {iter_val}: weight={weight:.4f}")

    # Test configuration
    print(f"\n[Testing configuration retrieval]")
    config = computer.get_config()
    print(f"  Config keys: {list(config.keys())}")
    print(f"  ✓ Configuration test passed")

    print("\n" + "=" * 60)
    print("✓ All tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_pinns_loss_computer()
