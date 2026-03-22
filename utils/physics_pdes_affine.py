"""
Physics-Informed Neural Networks (PINNs) - Affine Physics Constraints

This module implements 12-DOF affine transformation constraints for deformable objects.

Key differences from SE(3) rigid constraints:
- Replaces rigid_body constraint with strain_rate_limit
- Replaces divergence-free with volume_preservation (allows compression)
- Adds shear_limit and jacobian_determinant constraints

Author: FreeGave Team
Date: 2025
"""

import torch
import torch.nn as nn
from typing import Dict, Callable, Tuple
from utils.physics_pdes import PhysicsEquations


class AffinePhysicsEquations(PhysicsEquations):
    """
    Physics equations for 12-DOF affine transformations.

    Supports motion from rigid bodies to soft bodies:
    - Rigid: 6 DOF (SE(3))
    - Elastic: 12 DOF (affine with strain limits)
    - Soft: 12 DOF (affine with relaxed limits)
    """

    def __init__(self, device='cuda'):
        super(AffinePhysicsEquations, self).__init__(device)

    def strain_rate_residual(
        self,
        jacobian: torch.Tensor,
        max_strain_rate: float = 1.0
    ) -> torch.Tensor:
        """
        Strain rate constraint: limits the rate of deformation.

        Constraint: ||D||_F ≤ ε_max, where D = (∇v + ∇v^T) / 2

        Physical meaning:
        - D is the symmetric part of velocity gradient (strain rate tensor)
        - Prevents instantaneous infinite stretching/compression
        - Frobenius norm measures the magnitude of deformation

        Args:
            jacobian: [N, 3, 4] - velocity Jacobian ∂v/∂(x,y,z,t)
            max_strain_rate: maximum allowed strain rate (default: 1.0)

        Returns:
            residual: [N] - penalty for strain rates exceeding the limit
        """
        # Extract spatial gradient ∇v
        grad_v = jacobian[:, :, :3]  # [N, 3, 3]

        # Compute strain rate tensor D (symmetric part)
        D = (grad_v + grad_v.transpose(1, 2)) / 2.0  # [N, 3, 3]

        # Frobenius norm: ||D||_F = sqrt(Σ D_ij^2)
        strain_rate = torch.norm(D.reshape(D.shape[0], -1), dim=1)  # [N]

        # Soft constraint: only penalize excessive strain
        residual = torch.clamp(strain_rate - max_strain_rate, min=0.0)

        return residual

    def volume_preservation_residual(
        self,
        jacobian: torch.Tensor,
        compressibility: float = 0.1
    ) -> torch.Tensor:
        """
        Volume change constraint: allows slight compression/expansion.

        Constraint: |trace(D)| ≤ λ, where trace(D) = ∇·v

        Physical meaning:
        - trace(D) > 0: expansion (material dilates)
        - trace(D) < 0: compression (material contracts)
        - trace(D) = 0: incompressible (original divergence-free)

        This replaces the strict divergence-free constraint for deformable objects.

        Args:
            jacobian: [N, 3, 4] - velocity Jacobian
            compressibility: allowed volume change rate (default: 0.1 = 10%)

        Returns:
            residual: [N] - penalty for excessive volume change
        """
        # Extract spatial gradient
        grad_v = jacobian[:, :, :3]  # [N, 3, 3]

        # Compute strain rate tensor
        D = (grad_v + grad_v.transpose(1, 2)) / 2.0

        # Volume change rate = trace(D) = ∇·v
        volume_change_rate = torch.diagonal(D, dim1=1, dim2=2).sum(dim=1)  # [N]

        # Allow ±compressibility volume change
        residual = torch.clamp(volume_change_rate.abs() - compressibility, min=0.0)

        return residual

    def shear_limit_residual(
        self,
        jacobian: torch.Tensor,
        max_shear: float = 0.5
    ) -> torch.Tensor:
        """
        Shear constraint: limits shear deformation.

        Constraint: ||D_off|| ≤ γ_max, where D_off is off-diagonal part of D

        Physical meaning:
        - Off-diagonal elements of D represent shear strain rates
        - D_xy: shear in xy plane (like sliding cards)
        - Prevents excessive twisting/skewing

        Args:
            jacobian: [N, 3, 4] - velocity Jacobian
            max_shear: maximum allowed shear rate (default: 0.5)

        Returns:
            residual: [N] - penalty for excessive shear
        """
        # Extract spatial gradient
        grad_v = jacobian[:, :, :3]  # [N, 3, 3]

        # Compute strain rate tensor
        D = (grad_v + grad_v.transpose(1, 2)) / 2.0

        # Extract off-diagonal elements (shear components)
        D_off = D.clone()
        D_off[:, 0, 0] = 0  # Remove D_xx (normal strain)
        D_off[:, 1, 1] = 0  # Remove D_yy
        D_off[:, 2, 2] = 0  # Remove D_zz

        # Shear magnitude
        shear_magnitude = torch.norm(D_off.reshape(D.shape[0], -1), dim=1)  # [N]

        # Soft constraint
        residual = torch.clamp(shear_magnitude - max_shear, min=0.0)

        return residual

    def jacobian_determinant_residual(
        self,
        jacobian: torch.Tensor,
        min_det: float = 0.1,
        dt: float = 0.01
    ) -> torch.Tensor:
        """
        Jacobian determinant constraint: prevents material degeneracy.

        Constraint: det(I + dt·∇v) > δ_min

        Physical meaning:
        - det(F) = 0: volume collapse (material vanishes)
        - det(F) < 0: orientation flip (inside-out)
        - det(F) > 0: valid deformation

        This ensures the deformation mapping is invertible.

        Args:
            jacobian: [N, 3, 4] - velocity Jacobian
            min_det: minimum allowed determinant (default: 0.1)
            dt: time step for deformation gradient (default: 0.01)

        Returns:
            residual: [N] - penalty for near-degenerate deformations
        """
        # Extract spatial gradient
        grad_v_spatial = jacobian[:, :, :3]  # [N, 3, 3]

        # Deformation gradient: F = I + dt·∇v
        I = torch.eye(3, device=self.device).unsqueeze(0).expand(grad_v_spatial.shape[0], -1, -1)
        F = I + dt * grad_v_spatial  # [N, 3, 3]

        # Compute determinant
        det_F = torch.det(F)  # [N]

        # Penalize det < min_det (near collapse or flip)
        residual = torch.clamp(min_det - det_F, min=0.0)

        return residual

    def deviatoric_stress_residual(
        self,
        jacobian: torch.Tensor,
        max_deviatoric: float = 1.0
    ) -> torch.Tensor:
        """
        Deviatoric strain constraint: separates volume change from shape change.

        Constraint: ||D - (1/3)trace(D)·I||_F ≤ τ_max

        Physical meaning:
        - Deviatoric part: shape change without volume change
        - Useful for materials that resist shape change (like rubber)

        Args:
            jacobian: [N, 3, 4] - velocity Jacobian
            max_deviatoric: maximum deviatoric strain rate

        Returns:
            residual: [N] - penalty for excessive deviatoric strain
        """
        grad_v = jacobian[:, :, :3]
        D = (grad_v + grad_v.transpose(1, 2)) / 2.0

        # Volumetric part (hydrostatic)
        trace_D = torch.diagonal(D, dim1=1, dim2=2).sum(dim=1, keepdim=True)  # [N, 1]
        I = torch.eye(3, device=self.device).unsqueeze(0).expand(D.shape[0], -1, -1)
        volumetric_part = (trace_D.unsqueeze(-1) / 3.0) * I  # [N, 3, 3]

        # Deviatoric part
        D_dev = D - volumetric_part  # [N, 3, 3]

        # Frobenius norm
        deviatoric_magnitude = torch.norm(D_dev.reshape(D.shape[0], -1), dim=1)

        residual = torch.clamp(deviatoric_magnitude - max_deviatoric, min=0.0)

        return residual

    def compute_all_residuals(
        self,
        velocity_func: Callable[[torch.Tensor], torch.Tensor],
        acceleration_func: Callable[[torch.Tensor], torch.Tensor],
        xyzt: torch.Tensor,
        enable_transport: bool = True,
        enable_energy: bool = True,
        enable_strain_limit: bool = True,
        enable_volume_preservation: bool = True,
        enable_shear_limit: bool = False,
        enable_jacobian_det: bool = False,
        enable_deviatoric: bool = False,
        # Constraint parameters
        max_strain_rate: float = 1.0,
        compressibility: float = 0.1,
        max_shear: float = 0.5,
        min_det: float = 0.1,
        max_deviatoric: float = 1.0,
        dt: float = 0.01,
        chunk_size: int = 1000,
        debug_first_call: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all requested affine physics equation residuals.

        This is the main entry point, replacing PhysicsEquations.compute_all_residuals.

        Args:
            velocity_func: velocity computation function
            acceleration_func: acceleration computation function
            xyzt: [N, 4] - collocation points
            enable_*: flags to enable specific constraints
            max_strain_rate: parameter for strain_rate constraint
            compressibility: parameter for volume_preservation
            max_shear: parameter for shear_limit
            min_det: parameter for jacobian_det
            max_deviatoric: parameter for deviatoric constraint
            dt: time step for jacobian determinant
            chunk_size: batch size for Jacobian computation
            debug_first_call: enable debug output

        Returns:
            residuals: dict with constraint residuals
        """
        # Compute velocity, acceleration, and Jacobian
        velocity, acceleration, jacobian = self.compute_jacobian_and_values(
            velocity_func, acceleration_func, xyzt,
            chunk_size=chunk_size,
            debug_first_call=debug_first_call
        )

        residuals = {}

        # Universal constraints (from base class)
        if enable_transport:
            residuals['transport'] = self.transport_equation_residual(
                velocity, acceleration, jacobian
            )

        if enable_energy:
            residuals['energy'] = self.energy_conservation_residual(
                velocity, acceleration, jacobian
            )

        # Affine-specific constraints (new)
        if enable_strain_limit:
            residuals['strain_limit'] = self.strain_rate_residual(
                jacobian, max_strain_rate
            )

        if enable_volume_preservation:
            residuals['volume_preservation'] = self.volume_preservation_residual(
                jacobian, compressibility
            )

        if enable_shear_limit:
            residuals['shear_limit'] = self.shear_limit_residual(
                jacobian, max_shear
            )

        if enable_jacobian_det:
            residuals['jacobian_det'] = self.jacobian_determinant_residual(
                jacobian, min_det, dt
            )

        if enable_deviatoric:
            residuals['deviatoric'] = self.deviatoric_stress_residual(
                jacobian, max_deviatoric
            )

        # Clean up
        del velocity, acceleration, jacobian

        return residuals


def test_affine_physics_equations():
    """
    Unit test for affine physics equations.
    """
    print("=" * 60)
    print("Testing AffinePhysicsEquations Module")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    physics = AffinePhysicsEquations(device=device)

    print(f"\nDevice: {device}")

    # Create test data
    N = 100
    xyzt = torch.randn(N, 4, device=device, requires_grad=True)

    # Test velocity: affine deformation v = A·x where A is non-symmetric
    def test_velocity_func(xyzt_batch):
        """Affine velocity field with shear"""
        x, y, z = xyzt_batch[:, 0:1], xyzt_batch[:, 1:2], xyzt_batch[:, 2:3]
        vx = x + 0.5 * y  # shear in xy
        vy = y + 0.3 * z  # shear in yz
        vz = z + 0.2 * x  # shear in xz
        return torch.cat([vx, vy, vz], dim=1)

    def test_acceleration_func(xyzt_batch):
        return torch.ones(xyzt_batch.shape[0], 3, device=device) * 0.1

    # Import torch.func for Jacobian
    try:
        import torch.func as func
    except ImportError:
        print("torch.func not available, skipping tests")
        return

    # Compute Jacobian
    print("\n[1/6] Computing Jacobian for affine velocity field...")
    velocity, acceleration, jacobian = physics.compute_jacobian_and_values(
        test_velocity_func, test_acceleration_func, xyzt
    )
    print(f"  Velocity: {velocity.shape}")
    print(f"  Jacobian: {jacobian.shape}")

    # Test strain rate
    print("\n[2/6] Testing strain_rate_residual...")
    strain_res = physics.strain_rate_residual(jacobian, max_strain_rate=1.0)
    print(f"  Shape: {strain_res.shape}")
    print(f"  Mean: {strain_res.mean().item():.6f}")
    print(f"  Max: {strain_res.max().item():.6f}")
    assert strain_res.shape == (N,), "Strain residual shape mismatch"
    print("  ✓ Strain rate test passed")

    # Test volume preservation
    print("\n[3/6] Testing volume_preservation_residual...")
    vol_res = physics.volume_preservation_residual(jacobian, compressibility=0.1)
    print(f"  Shape: {vol_res.shape}")
    print(f"  Mean: {vol_res.mean().item():.6f}")
    assert vol_res.shape == (N,), "Volume residual shape mismatch"
    print("  ✓ Volume preservation test passed")

    # Test shear limit
    print("\n[4/6] Testing shear_limit_residual...")
    shear_res = physics.shear_limit_residual(jacobian, max_shear=0.5)
    print(f"  Shape: {shear_res.shape}")
    print(f"  Mean: {shear_res.mean().item():.6f}")
    assert shear_res.shape == (N,), "Shear residual shape mismatch"
    print("  ✓ Shear limit test passed")

    # Test Jacobian determinant
    print("\n[5/6] Testing jacobian_determinant_residual...")
    det_res = physics.jacobian_determinant_residual(jacobian, min_det=0.1, dt=0.01)
    print(f"  Shape: {det_res.shape}")
    print(f"  Mean: {det_res.mean().item():.6f}")
    assert det_res.shape == (N,), "Determinant residual shape mismatch"
    print("  ✓ Jacobian determinant test passed")

    # Test compute_all_residuals
    print("\n[6/6] Testing compute_all_residuals...")
    residuals = physics.compute_all_residuals(
        test_velocity_func,
        test_acceleration_func,
        xyzt,
        enable_transport=True,
        enable_energy=True,
        enable_strain_limit=True,
        enable_volume_preservation=True,
        enable_shear_limit=True,
        enable_jacobian_det=True
    )
    print(f"  Residuals: {list(residuals.keys())}")
    for key, val in residuals.items():
        print(f"    - {key}: shape={val.shape}, mean={val.abs().mean().item():.6f}")
    print("  ✓ Batch computation test passed")

    print("\n" + "=" * 60)
    print("✓ All affine physics tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_affine_physics_equations()
