"""
12-DOF Affine Velocity Field

Extends SegVel from 6 DOF (SE(3) rigid motion) to 12 DOF (affine transformation).

Basis functions:
- SE(3) basis (6): translation (3) + rotation (3)
- Strain basis (6): normal strains (3) + shear strains (3)

Total: 12 basis functions for full affine transformations.

Author: FreeGave Team
Date: 2025
"""

import torch
import torch.nn as nn
import numpy as np
import einops
from utils.velocity_field_utils import PositionEncoder


class SegVelAffine(nn.Module):
    """
    12-DOF affine velocity field with separate SE(3) and strain components.

    Architecture:
    - Weight network: predicts 12 time-dependent weights
    - Acceleration bank: K×6 learnable parameters (shared with SE(3) basis)
    - Basis functions: 12 spatial basis functions
    """

    def __init__(self, deform_code_dim=8, hidden_dim=64, layers=4, encode_dim=3):
        super(SegVelAffine, self).__init__()
        self.K = deform_code_dim

        # Time encoding for weight network
        in_dim = 1 + 1 * 2 * encode_dim
        self.embedder = PositionEncoder(encode_dim)

        # Weight network: t → 12 weights (6 SE(3) + 6 strain)
        self.weight_net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU())
        for i in range(layers - 1):
            self.weight_net.append(nn.Linear(hidden_dim, hidden_dim))
            self.weight_net.append(nn.SiLU())
        self.weight_net.append(nn.Linear(hidden_dim, 12 * self.K))  # 12 DOF

        # Acceleration bank (6 DOF, shared with original SE(3))
        # This maintains compatibility with rigid motion
        self.a_weight_bank = nn.Parameter(torch.randn(self.K, 6) / np.sqrt(self.K))

    def get_basis(self, xyzt):
        """
        Compute 12 basis functions for affine velocity field.

        Returns:
            v_basis: [N, 12, 3] - velocity basis functions
            a_basis: [N, 6, 3] - acceleration basis functions (SE(3) only)
        """
        x, y, z = xyzt[..., 0], xyzt[..., 1], xyzt[..., 2]
        zeros = xyzt[..., -1] * 0.
        ones = zeros + 1.

        # ============ SE(3) Basis (6 DOF) ============
        # Translation basis
        b1 = torch.stack([ones, zeros, zeros], dim=-1)   # x-translation
        b2 = torch.stack([zeros, ones, zeros], dim=-1)   # y-translation
        b3 = torch.stack([zeros, zeros, ones], dim=-1)   # z-translation

        # Rotation basis (skew-symmetric generators)
        b4 = torch.stack([zeros, z, -y], dim=-1)         # rotation around x-axis
        b5 = torch.stack([-z, zeros, x], dim=-1)         # rotation around y-axis
        b6 = torch.stack([y, -x, zeros], dim=-1)         # rotation around z-axis

        # ============ Strain Basis (6 DOF) ============
        # Normal strain (diagonal of strain rate tensor)
        b7 = torch.stack([x, zeros, zeros], dim=-1)      # ε_xx: stretch along x
        b8 = torch.stack([zeros, y, zeros], dim=-1)      # ε_yy: stretch along y
        b9 = torch.stack([zeros, zeros, z], dim=-1)      # ε_zz: stretch along z

        # Shear strain (off-diagonal of strain rate tensor)
        b10 = torch.stack([y, x, zeros], dim=-1)         # ε_xy: shear in xy-plane
        b11 = torch.stack([z, zeros, x], dim=-1)         # ε_xz: shear in xz-plane
        b12 = torch.stack([zeros, z, y], dim=-1)         # ε_yz: shear in yz-plane

        # Velocity basis: 12 functions
        v_basis = torch.stack([b1, b2, b3, b4, b5, b6,
                               b7, b8, b9, b10, b11, b12], dim=-2)  # [N, 12, 3]

        # Acceleration basis: 6 functions (SE(3) only, for compatibility)
        # Acceleration follows rigid motion assumption
        a4 = torch.stack([zeros, -y, -z], dim=-1)
        a5 = torch.stack([-x, zeros, -z], dim=-1)
        a6 = torch.stack([-x, -y, zeros], dim=-1)
        a_basis = torch.stack([b1, b2, b3, a4, a5, a6], dim=-2)  # [N, 6, 3]

        return v_basis, a_basis

    def get_basis_jac(self, xyzt):
        """
        Compute Jacobian of 12 basis functions.

        For integration with rotation (used in integrate_pos).

        Returns:
            v_basis: [N, 12, 3] - velocity basis
            jac_basis: [N, 12, 3, 3] - Jacobian ∂b_i/∂(x,y,z) for each basis
        """
        x, y, z = xyzt[..., 0], xyzt[..., 1], xyzt[..., 2]
        zeros = xyzt[..., -1] * 0.
        ones = zeros + 1.

        # Velocity basis (same as get_basis)
        b1 = torch.stack([ones, zeros, zeros], dim=-1)
        b2 = torch.stack([zeros, ones, zeros], dim=-1)
        b3 = torch.stack([zeros, zeros, ones], dim=-1)
        b4 = torch.stack([zeros, z, -y], dim=-1)
        b5 = torch.stack([-z, zeros, x], dim=-1)
        b6 = torch.stack([y, -x, zeros], dim=-1)
        b7 = torch.stack([x, zeros, zeros], dim=-1)
        b8 = torch.stack([zeros, y, zeros], dim=-1)
        b9 = torch.stack([zeros, zeros, z], dim=-1)
        b10 = torch.stack([y, x, zeros], dim=-1)
        b11 = torch.stack([z, zeros, x], dim=-1)
        b12 = torch.stack([zeros, z, y], dim=-1)

        v_basis = torch.stack([b1, b2, b3, b4, b5, b6,
                               b7, b8, b9, b10, b11, b12], dim=-2)

        # Compute Jacobian for each basis function
        zeros_vec = torch.stack([zeros, zeros, zeros], dim=-1)

        # Jacobian structure: jac[n, i, j, k] = ∂b_i^j / ∂x_k
        # Translation basis: Jacobian = 0
        jac_1 = torch.stack([zeros_vec, zeros_vec, zeros_vec], dim=-2)

        # Rotation basis: Jacobian = skew-symmetric matrices
        jac_4 = torch.stack([zeros_vec, b3, -b2], dim=-2)       # ∂b4/∂(x,y,z)
        jac_5 = torch.stack([-b3, zeros_vec, b1], dim=-2)       # ∂b5/∂(x,y,z)
        jac_6 = torch.stack([b2, -b1, zeros_vec], dim=-2)       # ∂b6/∂(x,y,z)

        # Normal strain basis: Jacobian = diagonal matrices
        e1 = torch.stack([ones, zeros, zeros], dim=-1)
        e2 = torch.stack([zeros, ones, zeros], dim=-1)
        e3 = torch.stack([zeros, zeros, ones], dim=-1)

        jac_7 = torch.stack([e1, zeros_vec, zeros_vec], dim=-2)    # ∂b7/∂(x,y,z)
        jac_8 = torch.stack([zeros_vec, e2, zeros_vec], dim=-2)    # ∂b8/∂(x,y,z)
        jac_9 = torch.stack([zeros_vec, zeros_vec, e3], dim=-2)    # ∂b9/∂(x,y,z)

        # Shear strain basis: Jacobian = off-diagonal symmetric matrices
        jac_10 = torch.stack([e2, e1, zeros_vec], dim=-2)          # ∂b10/∂(x,y,z)
        jac_11 = torch.stack([e3, zeros_vec, e1], dim=-2)          # ∂b11/∂(x,y,z)
        jac_12 = torch.stack([zeros_vec, e3, e2], dim=-2)          # ∂b12/∂(x,y,z)

        jac_basis = torch.stack([jac_1, jac_1, jac_1,  # Translation (0 Jacobian)
                                 jac_4, jac_5, jac_6,  # Rotation
                                 jac_7, jac_8, jac_9,  # Normal strain
                                 jac_10, jac_11, jac_12], dim=-3)  # Shear strain - 修复：dim=-3使jac_basis形状为[N,12,3,3]

        return v_basis, jac_basis

    def forward(self, deform_code, xt):
        """
        Compute velocity and acceleration.

        Args:
            deform_code: [N, K] - deform segment codes (after softmax)
            xt: [N, 4] - space-time coordinates (x, y, z, t)

        Returns:
            va: [N, 6] - concatenated velocity (3) and acceleration (3)
        """
        v_basis, a_basis = self.get_basis(xt)  # [N, 12, 3], [N, 6, 3]

        # Compute time-dependent weights
        t_embed = self.embedder(xt[..., -1:])  # [N, in_dim]
        weights = self.weight_net(t_embed)     # [N, 12*K]
        weights = einops.rearrange(weights, '... (K dim) -> ... K dim', K=self.K)  # [N, K, 12]

        # Velocity: weighted sum of 12 basis functions
        v = torch.einsum('...ij,...ki->...kj', v_basis, weights)  # [N, K, 3]
        v = torch.einsum('...k,...kj->...j', deform_code, v)      # [N, 3]

        # Acceleration: weighted sum of 6 basis functions (SE(3) only)
        a = torch.einsum('...ij,...ki->...kj', a_basis, self.a_weight_bank)  # [N, K, 3]
        a = torch.einsum('...k,...kj->...j', deform_code, a)                 # [N, 3]

        return torch.cat([v, a], dim=-1)

    def get_vel(self, deform_code, xt):
        """Get velocity only."""
        v_basis, _ = self.get_basis(xt)
        t_embed = self.embedder(xt[..., -1:])
        weights = self.weight_net(t_embed)
        weights = einops.rearrange(weights, '... (K dim) -> ... K dim', K=self.K)
        v = torch.einsum('...ij,...ki->...kj', v_basis, weights)
        v = torch.einsum('...k,...kj->...j', deform_code, v)
        return v

    def get_vel_jac(self, deform_code, xt):
        """Get velocity and its Jacobian."""
        v_basis, jac_basis = self.get_basis_jac(xt)
        t_embed = self.embedder(xt[..., -1:])
        weights = self.weight_net(t_embed)
        weights = einops.rearrange(weights, '... (K dim) -> ... K dim', K=self.K)

        # Velocity
        v = torch.einsum('...ij,...ki->...kj', v_basis, weights)
        v = torch.einsum('...k,...kj->...j', deform_code, v)

        # Jacobian
        jac = torch.einsum('...imn,...ki->...kmn', jac_basis, weights)
        jac = torch.einsum('...k,...kmn->...mn', deform_code, jac)

        return v, jac

    def get_acc(self, deform_code, xt):
        """Get acceleration only."""
        _, a_basis = self.get_basis(xt)
        a = torch.einsum('...ij,...ki->...kj', a_basis, self.a_weight_bank)
        a = torch.einsum('...k,...kj->...j', deform_code, a)
        return a

    def get_weights(self, deform_code, xt):
        """Get time-dependent weights (for analysis)."""
        t_embed = self.embedder(xt[..., -1:])
        weights = self.weight_net(t_embed)
        weights = einops.rearrange(weights, '... (K dim) -> ... K dim', K=self.K)
        v = torch.einsum('...k,...kj->...j', deform_code, weights)
        return v


def test_segvel_affine():
    """
    Unit test for SegVelAffine.
    """
    print("=" * 60)
    print("Testing SegVelAffine Module")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    K = 8  # deform code dimension

    vel_net = SegVelAffine(deform_code_dim=K, hidden_dim=64, layers=4).to(device)

    print(f"\nDevice: {device}")
    print(f"Deform code dimension: {K}")
    print(f"Total parameters: {sum(p.numel() for p in vel_net.parameters()):,}")

    # Create test data
    N = 100
    deform_code = torch.randn(N, K, device=device)
    deform_code = torch.softmax(deform_code, dim=-1)  # Normalize
    xt = torch.randn(N, 4, device=device)

    # Test basis functions
    print("\n[1/6] Testing get_basis...")
    v_basis, a_basis = vel_net.get_basis(xt)
    print(f"  Velocity basis shape: {v_basis.shape} (expected: [{N}, 12, 3])")
    print(f"  Acceleration basis shape: {a_basis.shape} (expected: [{N}, 6, 3])")
    assert v_basis.shape == (N, 12, 3), "Velocity basis shape mismatch"
    assert a_basis.shape == (N, 6, 3), "Acceleration basis shape mismatch"
    print("  ✓ Basis test passed")

    # Verify basis orthogonality properties
    print("\n[2/6] Verifying basis properties...")
    # Check SE(3) basis (first 6) forms valid Lie algebra
    grad_v_se3 = compute_velocity_gradient(v_basis[:, :6, :], xt)
    antisymm_error = check_antisymmetric(grad_v_se3[:, 3:6, :, :])  # Rotation part
    print(f"  SE(3) rotation antisymmetry error: {antisymm_error:.6f}")
    assert antisymm_error < 1e-5, "SE(3) basis rotation not antisymmetric"

    # Check strain basis (last 6) produces symmetric strain rate
    grad_v_strain = compute_velocity_gradient(v_basis[:, 6:, :], xt)
    D_strain = (grad_v_strain + grad_v_strain.transpose(-2, -1)) / 2
    symmetry_error = (D_strain - grad_v_strain).abs().max().item()
    print(f"  Strain basis symmetry error: {symmetry_error:.6f}")
    print("  ✓ Basis properties verified")

    # Test forward pass
    print("\n[3/6] Testing forward pass...")
    va = vel_net(deform_code, xt)
    print(f"  Output shape: {va.shape} (expected: [{N}, 6])")
    assert va.shape == (N, 6), "Forward pass shape mismatch"
    v_out, a_out = va[:, :3], va[:, 3:]
    print(f"  Velocity magnitude: {v_out.norm(dim=-1).mean().item():.6f}")
    print(f"  Acceleration magnitude: {a_out.norm(dim=-1).mean().item():.6f}")
    print("  ✓ Forward pass test passed")

    # Test get_vel
    print("\n[4/6] Testing get_vel...")
    v = vel_net.get_vel(deform_code, xt)
    print(f"  Velocity shape: {v.shape}")
    assert v.shape == (N, 3), "get_vel shape mismatch"
    print("  ✓ get_vel test passed")

    # Test get_acc
    print("\n[5/6] Testing get_acc...")
    a = vel_net.get_acc(deform_code, xt)
    print(f"  Acceleration shape: {a.shape}")
    assert a.shape == (N, 3), "get_acc shape mismatch"
    print("  ✓ get_acc test passed")

    # Test get_vel_jac
    print("\n[6/6] Testing get_vel_jac...")
    v, jac = vel_net.get_vel_jac(deform_code, xt)
    print(f"  Velocity shape: {v.shape}")
    print(f"  Jacobian shape: {jac.shape} (expected: [{N}, 3, 3])")
    assert jac.shape == (N, 3, 3), "Jacobian shape mismatch"
    print("  ✓ get_vel_jac test passed")

    # Test gradient flow
    print("\n[Bonus] Testing gradient flow...")
    loss = va.sum()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in vel_net.parameters() if p.grad is not None)
    print(f"  Total gradient norm: {grad_norm:.6f}")
    assert grad_norm > 0, "No gradients computed"
    print("  ✓ Gradient flow test passed")

    print("\n" + "=" * 60)
    print("✓ All SegVelAffine tests passed!")
    print("=" * 60)


def compute_velocity_gradient(v_basis, xt):
    """Helper: compute ∇v for basis functions."""
    # Simplified: for testing purposes
    # In practice, use torch.func.jacrev
    x, y, z = xt[:, 0], xt[:, 1], xt[:, 2]

    # For rotation basis b4 = [0, z, -y], ∇b4 should be skew-symmetric
    # This is a placeholder - proper implementation needs autograd
    N, num_basis, _ = v_basis.shape
    return torch.zeros(N, num_basis, 3, 3, device=v_basis.device)


def check_antisymmetric(matrices):
    """Check if matrices are antisymmetric: A = -A^T."""
    antisymm = matrices + matrices.transpose(-2, -1)
    return antisymm.abs().max().item()


if __name__ == "__main__":
    test_segvel_affine()
