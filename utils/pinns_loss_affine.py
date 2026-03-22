"""
Physics-Informed Neural Networks (PINNs) - Affine Loss Computation Module

This module extends PINNsLossComputer to support 12-DOF affine transformations.

Key changes from original:
- Replaces rigid_body and divergence with affine-specific constraints
- Adds material-specific preset configurations
- Implements curriculum learning for stiffness relaxation

Author: FreeGave Team
Date: 2025
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from utils.physics_pdes_affine import AffinePhysicsEquations
from utils.pinns_sampler import CollocationSampler


class AffinePINNsLossComputer(nn.Module):
    """
    PINNs loss computer for 12-DOF affine transformations.

    Supports material types from rigid to soft:
    - rigid: effectively 6 DOF (SE(3))
    - elastic: 12 DOF with moderate constraints
    - soft_body: 12 DOF with relaxed constraints
    - fluid: 12 DOF with incompressibility
    """

    # Preset configurations for different material types
    MATERIAL_PRESETS = {
        'rigid': {
            # Effectively SE(3): very strict strain/shear limits
            'enable_transport': True,
            'enable_energy': True,
            'enable_strain_limit': True,
            'enable_volume_preservation': True,
            'enable_shear_limit': True,
            'enable_jacobian_det': True,
            'enable_deviatoric': False,
            'weights': {
                'transport': 1.0,
                'energy': 0.1,
                'strain_limit': 10.0,        # Very strict
                'volume_preservation': 5.0,   # Near incompressible
                'shear_limit': 10.0,          # Very strict
                'jacobian_det': 1.0,
            },
            'params': {
                'max_strain_rate': 0.1,      # Almost no strain allowed
                'compressibility': 0.01,
                'max_shear': 0.05,
                'min_det': 0.1,
            }
        },

        'elastic': {
            # Moderate deformation (rubber, skin)
            'enable_transport': True,
            'enable_energy': True,
            'enable_strain_limit': True,
            'enable_volume_preservation': True,
            'enable_shear_limit': True,
            'enable_jacobian_det': True,
            'enable_deviatoric': False,
            'weights': {
                'transport': 1.0,
                'energy': 0.1,
                'strain_limit': 1.0,
                'volume_preservation': 0.5,
                'shear_limit': 0.5,
                'jacobian_det': 0.5,
            },
            'params': {
                'max_strain_rate': 1.0,
                'compressibility': 0.1,       # 10% volume change allowed
                'max_shear': 0.5,
                'min_det': 0.1,
            }
        },

        'soft_body': {
            # Large deformation (cloth, gelatin)
            'enable_transport': True,
            'enable_energy': True,
            'enable_strain_limit': True,
            'enable_volume_preservation': True,
            'enable_shear_limit': False,       # Free shear
            'enable_jacobian_det': True,
            'enable_deviatoric': False,
            'weights': {
                'transport': 1.0,
                'energy': 0.1,
                'strain_limit': 0.1,
                'volume_preservation': 0.5,
                'jacobian_det': 0.5,
            },
            'params': {
                'max_strain_rate': 3.0,        # Large deformation allowed
                'compressibility': 0.2,
                'max_shear': 1.0,              # Not used, but kept for completeness
                'min_det': 0.05,
            }
        },

        'fluid': {
            # Incompressible flow
            'enable_transport': True,
            'enable_energy': True,
            'enable_strain_limit': False,      # Free deformation rate
            'enable_volume_preservation': True,
            'enable_shear_limit': False,       # Free shear
            'enable_jacobian_det': False,
            'enable_deviatoric': False,
            'weights': {
                'transport': 1.0,
                'energy': 0.1,
                'volume_preservation': 2.0,    # Strong incompressibility
            },
            'params': {
                'compressibility': 0.01,       # Nearly incompressible
            }
        },

        'custom': {
            # Fully customizable
            'enable_transport': True,
            'enable_energy': True,
            'enable_strain_limit': True,
            'enable_volume_preservation': True,
            'enable_shear_limit': True,
            'enable_jacobian_det': True,
            'enable_deviatoric': False,
            'weights': {
                'transport': 1.0,
                'energy': 0.1,
                'strain_limit': 1.0,
                'volume_preservation': 0.5,
                'shear_limit': 0.5,
                'jacobian_det': 0.5,
            },
            'params': {
                'max_strain_rate': 1.0,
                'compressibility': 0.1,
                'max_shear': 0.5,
                'min_det': 0.1,
            }
        }
    }

    def __init__(
        self,
        material_type: str = 'elastic',
        pinns_start_iter: int = 3000,
        curriculum_end_iter: int = 40000,
        max_pinns_weight: float = 0.1,
        device: str = 'cuda',
        # Custom overrides (only used if material_type='custom' or for fine-tuning)
        weight_transport: float = None,
        weight_energy: float = None,
        weight_strain_limit: float = None,
        weight_volume_preservation: float = None,
        weight_shear_limit: float = None,
        weight_jacobian_det: float = None,
        max_strain_rate: float = None,
        compressibility: float = None,
        max_shear: float = None,
        min_det: float = None,
    ):
        """
        Initialize affine PINNs loss computer.

        Args:
            material_type: preset material ('rigid', 'elastic', 'soft_body', 'fluid', 'custom')
            pinns_start_iter: iteration to start PINNs training
            curriculum_end_iter: iteration to reach max weight
            max_pinns_weight: maximum curriculum weight
            device: computation device
            weight_*: custom weight overrides
            max_strain_rate, compressibility, max_shear, min_det: constraint parameters
        """
        super(AffinePINNsLossComputer, self).__init__()

        self.material_type = material_type
        self.pinns_start_iter = pinns_start_iter
        self.curriculum_end_iter = curriculum_end_iter
        self.max_pinns_weight = max_pinns_weight
        self.device = device

        # Load preset configuration
        if material_type not in self.MATERIAL_PRESETS:
            raise ValueError(f"Unknown material_type: {material_type}. "
                           f"Available: {list(self.MATERIAL_PRESETS.keys())}")

        preset = self.MATERIAL_PRESETS[material_type]
        self.equation_flags = {k: v for k, v in preset.items() if k.startswith('enable_')}
        self.equation_weights = preset['weights'].copy()
        self.constraint_params = preset['params'].copy()

        # Apply custom overrides
        if weight_transport is not None:
            self.equation_weights['transport'] = weight_transport
        if weight_energy is not None:
            self.equation_weights['energy'] = weight_energy
        if weight_strain_limit is not None:
            self.equation_weights['strain_limit'] = weight_strain_limit
        if weight_volume_preservation is not None:
            self.equation_weights['volume_preservation'] = weight_volume_preservation
        if weight_shear_limit is not None:
            self.equation_weights['shear_limit'] = weight_shear_limit
        if weight_jacobian_det is not None:
            self.equation_weights['jacobian_det'] = weight_jacobian_det

        if max_strain_rate is not None:
            self.constraint_params['max_strain_rate'] = max_strain_rate
        if compressibility is not None:
            self.constraint_params['compressibility'] = compressibility
        if max_shear is not None:
            self.constraint_params['max_shear'] = max_shear
        if min_det is not None:
            self.constraint_params['min_det'] = min_det

        # Initialize physics equations module (affine version)
        self.physics_eqs = AffinePhysicsEquations(device=device)

        # Initialize sampler (same as original)
        self.sampler = CollocationSampler(device=device)

    def get_curriculum_weight(self, iteration: int) -> float:
        """
        Get curriculum weight with optional stiffness relaxation.

        Args:
            iteration: current training iteration

        Returns:
            weight: curriculum weight [0, max_pinns_weight]
        """
        if iteration < self.pinns_start_iter:
            return 0.0
        elif iteration >= self.curriculum_end_iter:
            return self.max_pinns_weight
        else:
            progress = (iteration - self.pinns_start_iter) / (
                self.curriculum_end_iter - self.pinns_start_iter)
            return self.max_pinns_weight * progress

    def create_velocity_func(self, deform_model, deform_code: torch.Tensor):
        """
        Create velocity function for Jacobian computation.

        Args:
            deform_model: DeformModel with vel_net
            deform_code: [M, K] - deform codes for selected points

        Returns:
            velocity_func: callable for physics equations
        """
        # Get segmented deform code
        deform_seg_full = deform_model.code_field.seg(deform_code).detach()
        vel_net = deform_model.vel_net

        def velocity_func_batch(xyzt: torch.Tensor) -> torch.Tensor:
            return vel_net.get_vel(deform_seg_full, xyzt)

        # Attach attributes for physics_pdes
        velocity_func_batch.deform_seg = deform_seg_full
        velocity_func_batch.vel_net = vel_net

        return velocity_func_batch

    def create_acceleration_func(self, deform_model, deform_code: torch.Tensor):
        """
        Create acceleration function.

        Args:
            deform_model: DeformModel with vel_net
            deform_code: [M, K] - deform codes for selected points

        Returns:
            acceleration_func: callable for physics equations
        """
        deform_seg_full = deform_model.code_field.seg(deform_code).detach()
        vel_net = deform_model.vel_net

        def acceleration_func_batch(xyzt: torch.Tensor) -> torch.Tensor:
            return vel_net.get_acc(deform_seg_full, xyzt)

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
        Compute affine PINNs loss.

        Args:
            deform_model: DeformModel instance
            gaussians: GaussianModel instance
            t: current time (scalar tensor)
            iteration: current training iteration

        Returns:
            total_loss: weighted PINNs loss
            loss_dict: individual loss components for logging
        """
        # Check curriculum weight
        curriculum_weight = self.get_curriculum_weight(iteration)
        if curriculum_weight == 0.0:
            return torch.tensor(0.0, device=self.device), {}

        # Get data (detached for efficiency)
        with torch.no_grad():
            xyz = gaussians.get_xyz.detach()
            deform_code = deform_model.code_field(xyz).detach()

        # Sample collocation points
        collocation_points, selected_indices = self.sampler.sample_collocation_points(
            xyz, t, iteration
        )

        # Get selected deform codes
        with torch.no_grad():
            selected_deform_code = deform_code[selected_indices].detach()

        # Create velocity/acceleration functions
        velocity_func = self.create_velocity_func(deform_model, selected_deform_code)
        acceleration_func = self.create_acceleration_func(deform_model, selected_deform_code)

        # Debug output for first PINNs call
        debug_first_call = (iteration == self.pinns_start_iter)
        if debug_first_call:
            print(f"  [Affine PINNs] Material type: {self.material_type}")
            print(f"  [Affine PINNs] Enabled constraints: {[k for k, v in self.equation_flags.items() if v]}")

        # Compute residuals using affine physics equations
        residuals = self.physics_eqs.compute_all_residuals(
            velocity_func,
            acceleration_func,
            collocation_points,
            **self.equation_flags,
            **self.constraint_params,
            debug_first_call=debug_first_call
        )

        # Compute weighted losses
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=self.device)

        for eq_name, residual in residuals.items():
            weight = self.equation_weights.get(eq_name, 0.0)
            if weight > 0:
                eq_loss = torch.mean(residual ** 2)
                weighted_loss = weight * eq_loss
                total_loss = total_loss + weighted_loss
                loss_dict[f'pinns_{eq_name}'] = eq_loss.item()

        # Apply curriculum weight
        total_loss = curriculum_weight * total_loss
        loss_dict['pinns_total'] = total_loss.item()
        loss_dict['pinns_curriculum_weight'] = curriculum_weight
        loss_dict['pinns_material_type'] = self.material_type

        # Cleanup
        del velocity_func, acceleration_func, residuals
        del selected_deform_code, collocation_points

        return total_loss, loss_dict

    def get_config(self) -> dict:
        """Get current configuration for logging."""
        return {
            'material_type': self.material_type,
            'equation_flags': self.equation_flags,
            'equation_weights': self.equation_weights,
            'constraint_params': self.constraint_params,
            'pinns_start_iter': self.pinns_start_iter,
            'curriculum_end_iter': self.curriculum_end_iter,
            'max_pinns_weight': self.max_pinns_weight,
        }


def test_affine_pinns_loss_computer():
    """
    Unit test for AffinePINNsLossComputer.
    """
    print("=" * 60)
    print("Testing AffinePINNsLossComputer Module")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test different material types
    for material_type in ['rigid', 'elastic', 'soft_body', 'fluid']:
        print(f"\n[Testing material: {material_type}]")
        computer = AffinePINNsLossComputer(
            material_type=material_type,
            pinns_start_iter=3000,
            curriculum_end_iter=30000,
            max_pinns_weight=0.1,
            device=device
        )

        print(f"  Enabled constraints:")
        for key, val in computer.equation_flags.items():
            if val:
                print(f"    - {key}")

        print(f"  Weights: {computer.equation_weights}")
        print(f"  Params: {computer.constraint_params}")

        # Test curriculum schedule
        print(f"  Curriculum schedule:")
        for iter_val in [0, 3000, 16500, 30000]:
            weight = computer.get_curriculum_weight(iter_val)
            print(f"    Iteration {iter_val}: weight={weight:.4f}")

    # Test custom overrides
    print("\n[Testing custom overrides]")
    computer = AffinePINNsLossComputer(
        material_type='elastic',
        weight_strain_limit=2.0,   # Override
        max_strain_rate=0.5,       # Override
        device=device
    )
    print(f"  strain_limit weight: {computer.equation_weights['strain_limit']} (overridden to 2.0)")
    print(f"  max_strain_rate: {computer.constraint_params['max_strain_rate']} (overridden to 0.5)")

    # Test config
    config = computer.get_config()
    print(f"\n[Configuration retrieval]")
    print(f"  Config keys: {list(config.keys())}")
    print("  ✓ Configuration test passed")

    print("\n" + "=" * 60)
    print("✓ All AffinePINNsLossComputer tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_affine_pinns_loss_computer()
