"""
Physics-Informed Neural Networks (PINNs) - Physics Equations Module

This module implements physics equation residual calculations for constraining
the velocity field in the FreeGave dynamic scene reconstruction framework.

Core Physics Equations:
1. Transport Equation: ∂v/∂t + (v·∇)v = a
2. Divergence-Free: ∇·v = 0 (for incompressible flows)
3. Rigid Body Constraint: ∇v should be skew-symmetric for rigid motion
4. Energy Conservation: v·(∂v/∂t) = v·a

Author: FreeGave Team
Date: 2025
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Callable

# Import torch.func for efficient Jacobian computation
try:
    import torch.func as func
    HAS_TORCH_FUNC = True
except ImportError:
    HAS_TORCH_FUNC = False
    raise ImportError(
        "torch.func is required for PINNs. Please upgrade to PyTorch >= 2.0\n"
        "Install: pip install torch>=2.0.0"
    )


class PhysicsEquations(nn.Module):
    """
    Compute physics equation residuals for PINNs training.

    Uses torch.func.jacrev() for optimal performance (requires PyTorch 2.0+).
    """

    def __init__(self, device='cuda'):
        super(PhysicsEquations, self).__init__()
        self.device = device

    def compute_jacobian_and_values(
        self,
        velocity_func: Callable[[torch.Tensor], torch.Tensor],
        acceleration_func: Callable[[torch.Tensor], torch.Tensor],
        xyzt: torch.Tensor,
        chunk_size: int = 100,  # ← 小chunk避免OOM（配合1-2%采样率）
        debug_first_call: bool = False  # ← 添加调试标志
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute velocity, acceleration, and velocity Jacobian using torch.func.

        CRITICAL FIX for both OOM and dimension errors:
        - Use small chunks (100 points) to prevent 509GB allocation
        - Direct slicing (no detach) to preserve gradient and dimensions
        - Maintain correct [N, 3, 4] jacobian shape

        Args:
            velocity_func: Function (xyzt: [N, 4]) -> velocity: [N, 3]
            acceleration_func: Function (xyzt: [N, 4]) -> acceleration: [N, 3]
            xyzt: [N, 4] - collocation points (x, y, z, t)
            chunk_size: maximum number of points to process at once (default: 100)

        Returns:
            velocity: [N, 3]
            acceleration: [N, 3]
            jacobian: [N, 3, 4] - ∂v/∂(x,y,z,t)
        """
        N = xyzt.shape[0]

        if debug_first_call:
            print(f"    [physics_pdes] compute_jacobian_and_values: N={N}, chunk_size={chunk_size}")

        # 关键调试：打印实际输入形状
        # print(f"[DEBUG physics_pdes] xyzt shape: {xyzt.shape}, dtype: {xyzt.dtype}, device: {xyzt.device}")
        # print(f"[DEBUG physics_pdes] xyzt requires_grad: {xyzt.requires_grad}")

        # 调试：验证输入形状
        if xyzt.dim() != 2 or xyzt.shape[1] != 4:
            raise ValueError(f"[ERROR] Expected xyzt shape [N, 4], got {xyzt.shape}")

        # 如果点数很少，直接计算（无需分块）
        if N <= chunk_size:
            # print(f"[DEBUG physics_pdes] Single batch (N={N} <= {chunk_size})")
            if debug_first_call:
                print(f"    [physics_pdes] Single batch mode (N <= chunk_size)")
            result = self._compute_jacobian_single_batch(
                velocity_func, acceleration_func, xyzt, debug_first_call
            )
            # vel, acc, jac = result
            # print(f"[DEBUG physics_pdes] Result shapes: vel={vel.shape}, acc={acc.shape}, jac={jac.shape}")
            return result

        # 分块策略：小chunk + 直接切片（保持梯度和形状）
        # print(f"[DEBUG physics_pdes] Chunked processing: N={N}, chunk_size={chunk_size}")
        if debug_first_call:
            print(f"    [physics_pdes] Chunked mode: {(N + chunk_size - 1) // chunk_size} chunks")

        # 关键修复：提取deform_seg用于分块（如果存在）
        has_deform_seg = hasattr(velocity_func, 'deform_seg')
        deform_seg_full = velocity_func.deform_seg if has_deform_seg else None

        velocities = []
        jacobians = []
        accelerations = []

        for i in range(0, N, chunk_size):
            end_idx = min(i + chunk_size, N)
            # 直接切片，保持形状[chunk, 4]和梯度连接
            xyzt_chunk = xyzt[i:end_idx]
            # print(f"[DEBUG physics_pdes] Chunk {i//chunk_size}: xyzt_chunk shape={xyzt_chunk.shape}")

            if debug_first_call and i == 0:
                print(f"    [physics_pdes] Processing first chunk...")

            # 关键修复：如果有deform_seg，需要创建同步切分的临时函数
            if has_deform_seg:
                deform_seg_chunk = deform_seg_full[i:end_idx]
                # 创建临时函数，使用切分后的deform_seg
                vel_net = velocity_func.vel_net

                def velocity_func_chunk(xyzt):
                    return vel_net.get_vel(deform_seg_chunk, xyzt)

                def acceleration_func_chunk(xyzt):
                    return vel_net.get_acc(deform_seg_chunk, xyzt)

                # 给临时函数添加属性
                velocity_func_chunk.deform_seg = deform_seg_chunk
                velocity_func_chunk.vel_net = vel_net

                vel, acc, jac = self._compute_jacobian_single_batch(
                    velocity_func_chunk, acceleration_func_chunk, xyzt_chunk,
                    debug_first_call=(debug_first_call and i==0)
                )
            else:
                # 旧方式：直接使用原始函数
                vel, acc, jac = self._compute_jacobian_single_batch(
                    velocity_func, acceleration_func, xyzt_chunk,
                    debug_first_call=(debug_first_call and i==0)
                )
            # print(f"[DEBUG physics_pdes] Chunk result: vel={vel.shape}, jac={jac.shape}")

            velocities.append(vel)
            accelerations.append(acc)
            jacobians.append(jac)

        # 合并结果（保持梯度流）
        velocity = torch.cat(velocities, dim=0)
        acceleration = torch.cat(accelerations, dim=0)
        jacobian = torch.cat(jacobians, dim=0)

        # print(f"[DEBUG physics_pdes] Final shapes: velocity={velocity.shape}, jacobian={jacobian.shape}")

        # 最终验证输出形状
        if jacobian.shape != (N, 3, 4):
            raise ValueError(f"[ERROR] Expected jacobian shape [{N}, 3, 4], got {jacobian.shape}")

        return velocity, acceleration, jacobian

    def _compute_jacobian_single_batch(
        self,
        velocity_func: Callable[[torch.Tensor], torch.Tensor],
        acceleration_func: Callable[[torch.Tensor], torch.Tensor],
        xyzt: torch.Tensor,
        debug_first_call: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Jacobian for a single batch (internal helper).

        CRITICAL FIX: velocity_func captures batch-level deform_seg [N, K].
        We need to use vmap with proper indexing.
        """
        N = xyzt.shape[0]

        if debug_first_call:
            print(f"    [physics_pdes] _compute_jacobian_single_batch: computing velocity and acceleration...")

        # 批量计算velocity和acceleration
        velocity = velocity_func(xyzt)  # [N, 3]
        acceleration = acceleration_func(xyzt)  # [N, 3]

        if debug_first_call:
            print(f"    [physics_pdes] velocity shape: {velocity.shape}, acceleration shape: {acceleration.shape}")

        # print(f"[DEBUG _compute_jacobian_single_batch] velocity shape: {velocity.shape}")
        # print(f"[DEBUG _compute_jacobian_single_batch] N: {N}")

        # 检查velocity_func是否有deform_seg属性
        if hasattr(velocity_func, 'deform_seg') and hasattr(velocity_func, 'vel_net'):
            # 新方式：使用vmap同时处理deform_seg和xyzt
            deform_seg = velocity_func.deform_seg  # [N, K]
            vel_net = velocity_func.vel_net

            if debug_first_call:
                print(f"    [physics_pdes] Using vmap with deform_seg shape: {deform_seg.shape}")
                print(f"    [physics_pdes] Computing Jacobian via torch.func.vmap...")

            # print(f"[DEBUG] Using vmap with deform_seg shape: {deform_seg.shape}")

            # 定义单点速度函数（接受单个deform_seg和单个xyzt）
            def single_point_vel(single_seg, single_xyzt):
                # single_seg: [K], single_xyzt: [4] -> [3]
                return vel_net.get_vel(single_seg.unsqueeze(0), single_xyzt.unsqueeze(0)).squeeze(0)

            # 使用vmap同时向量化deform_seg和xyzt
            jacobian_fn = func.jacrev(single_point_vel, argnums=1)  # 对第2个参数(xyzt)求导
            jacobian = func.vmap(jacobian_fn)(deform_seg, xyzt)  # [N, 3, 4]

            if debug_first_call:
                print(f"    [physics_pdes] Jacobian computed successfully, shape: {jacobian.shape}")

            # print(f"[DEBUG] Jacobian computed with shape: {jacobian.shape}")

        else:
            # 旧方式：假设velocity_func可以处理单点输入（调试脚本测试用）
            # print(f"[WARN] velocity_func doesn't have deform_seg attribute, using old method")

            def single_velocity(single_xyzt):
                return velocity_func(single_xyzt.unsqueeze(0)).squeeze(0)

            jacobian_fn = func.jacrev(single_velocity)
            jacobian = func.vmap(jacobian_fn)(xyzt)  # [N, 3, 4]

        return velocity, acceleration, jacobian

    def transport_equation_residual(
        self,
        velocity: torch.Tensor,
        acceleration: torch.Tensor,
        jacobian: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute transport equation residual: ∂v/∂t + (v·∇)v - a

        This is the Navier-Stokes advection term without pressure/viscosity.

        Args:
            velocity: [N, 3] - velocity at collocation points
            acceleration: [N, 3] - acceleration at collocation points
            jacobian: [N, 3, 4] - Jacobian ∂v/∂(x,y,z,t)

        Returns:
            residual: [N, 3] - transport equation residual
        """
        # Extract spatial gradient: ∇v = ∂v/∂(x,y,z)
        grad_v_spatial = jacobian[:, :, :3]  # [N, 3, 3]
        # jacobian[n, i, j] = ∂v_i/∂x_j

        # Extract temporal derivative: ∂v/∂t
        dv_dt = jacobian[:, :, 3]  # [N, 3]

        # Compute convective term: (v·∇)v_j = Σ_i v_i * ∂v_j/∂x_i
        # We need ∂v_j/∂x_i = jacobian[n, j, i] = grad_v_spatial.transpose(-2, -1)[n, i, j]
        # Fixed: changed 'nij' to 'nji' to compute correct convective term
        convective_term = torch.einsum('ni,nji->nj', velocity, grad_v_spatial)

        # Residual: ∂v/∂t + (v·∇)v - a
        residual = dv_dt + convective_term - acceleration

        return residual

    def divergence_residual(
        self,
        jacobian: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute divergence residual: ∇·v

        For incompressible flows (fluids), divergence should be zero.

        Args:
            jacobian: [N, 3, 4] - Jacobian ∂v/∂(x,y,z,t)

        Returns:
            residual: [N] - divergence values
        """
        # Divergence: ∇·v = ∂v_x/∂x + ∂v_y/∂y + ∂v_z/∂z
        divergence = jacobian[:, 0, 0] + jacobian[:, 1, 1] + jacobian[:, 2, 2]

        return divergence

    def rigid_body_residual(
        self,
        jacobian: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute rigid body constraint residual.

        For rigid body motion, velocity gradient should be skew-symmetric:
        (∇v + ∇v^T) = 0

        Args:
            jacobian: [N, 3, 4] - Jacobian ∂v/∂(x,y,z,t)

        Returns:
            residual: [N] - Frobenius norm of symmetric part
        """
        # Extract spatial Jacobian
        grad_v = jacobian[:, :, :3]  # [N, 3, 3]

        # Compute symmetric part: (∇v + ∇v^T) / 2
        symmetric_part = (grad_v + grad_v.transpose(1, 2)) / 2.0

        # Frobenius norm: ||symmetric_part||_F
        residual = torch.norm(symmetric_part.reshape(symmetric_part.shape[0], -1), dim=1)

        return residual

    def energy_conservation_residual(
        self,
        velocity: torch.Tensor,
        acceleration: torch.Tensor,
        jacobian: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute energy conservation residual.

        For conservative systems: d/dt(0.5*||v||²) = v·a
        This simplifies to: v·(∂v/∂t) - v·a = 0

        Args:
            velocity: [N, 3] - velocity at collocation points
            acceleration: [N, 3] - acceleration at collocation points
            jacobian: [N, 3, 4] - Jacobian ∂v/∂(x,y,z,t)

        Returns:
            residual: [N] - energy conservation residual
        """
        # Extract temporal derivative of velocity
        dv_dt = jacobian[:, :, 3]  # [N, 3]

        # Left side: v·(∂v/∂t)
        lhs = torch.sum(velocity * dv_dt, dim=1)

        # Right side: v·a
        rhs = torch.sum(velocity * acceleration, dim=1)

        # Residual
        residual = lhs - rhs

        return residual

    def compute_all_residuals(
        self,
        velocity_func: Callable[[torch.Tensor], torch.Tensor],
        acceleration_func: Callable[[torch.Tensor], torch.Tensor],
        xyzt: torch.Tensor,
        enable_transport: bool = True,
        enable_divergence: bool = False,
        enable_rigid_body: bool = False,
        enable_energy: bool = False,
        chunk_size: int = 1000,  # 添加chunk_size参数
        debug_first_call: bool = False  # 添加调试标志
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all requested physics equation residuals.

        This is the main entry point called by PINNsLossComputer.

        Args:
            velocity_func: Function that computes velocity from xyzt [N,4] -> [N,3]
            acceleration_func: Function that computes acceleration from xyzt [N,4] -> [N,3]
            xyzt: [N, 4] - collocation points (x,y,z,t)
            enable_transport: whether to compute transport equation residual
            enable_divergence: whether to compute divergence residual
            enable_rigid_body: whether to compute rigid body residual
            enable_energy: whether to compute energy conservation residual
            chunk_size: maximum points per batch (default: 1000)
            debug_first_call: whether to print debug info for first call

        Returns:
            residuals: dict with keys 'transport', 'divergence', 'rigid_body', 'energy'
        """
        # Compute velocity, acceleration, and Jacobian in one efficient pass
        velocity, acceleration, jacobian = self.compute_jacobian_and_values(
            velocity_func, acceleration_func, xyzt, chunk_size=chunk_size, debug_first_call=debug_first_call
        )

        residuals = {}

        if enable_transport:
            residuals['transport'] = self.transport_equation_residual(
                velocity, acceleration, jacobian
            )

        if enable_divergence:
            residuals['divergence'] = self.divergence_residual(jacobian)

        if enable_rigid_body:
            residuals['rigid_body'] = self.rigid_body_residual(jacobian)

        if enable_energy:
            residuals['energy'] = self.energy_conservation_residual(
                velocity, acceleration, jacobian
            )

        # 注意：不能detach residuals，否则梯度无法回传
        # 清理中间变量即可（Python会自动垃圾回收）
        del velocity, acceleration, jacobian

        return residuals


def test_physics_equations():
    """
    Unit test for physics equations module.
    """
    print("=" * 60)
    print("Testing PhysicsEquations Module (torch.func)")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    physics = PhysicsEquations(device=device)

    print(f"\nPyTorch version: {torch.__version__}")
    print(f"torch.func available: {HAS_TORCH_FUNC}")
    print(f"Device: {device}")

    # Create sample data
    N = 100
    xyzt = torch.randn(N, 4, device=device)

    print(f"Test data shape: N={N}")

    # Define simple velocity and acceleration functions for testing
    def test_velocity_func(xyzt_batch):
        """Simple test: v = [x+t, y+t, z+t]"""
        return xyzt_batch[:, :3] + xyzt_batch[:, 3:4]

    def test_acceleration_func(xyzt_batch):
        """Simple test: a = [1, 1, 1]"""
        return torch.ones(xyzt_batch.shape[0], 3, device=device)

    # Test Jacobian computation
    print("\n[1/5] Testing Jacobian computation with torch.func...")
    velocity, acceleration, jacobian = physics.compute_jacobian_and_values(
        test_velocity_func, test_acceleration_func, xyzt
    )
    print(f"  Velocity shape: {velocity.shape}")
    print(f"  Acceleration shape: {acceleration.shape}")
    print(f"  Jacobian shape: {jacobian.shape}")
    print(f"  Expected Jacobian shape: torch.Size([{N}, 3, 4])")
    assert velocity.shape == (N, 3), "Velocity shape mismatch!"
    assert acceleration.shape == (N, 3), "Acceleration shape mismatch!"
    assert jacobian.shape == (N, 3, 4), "Jacobian shape mismatch!"
    print(f"  ✓ Jacobian computation test passed")

    # Verify Jacobian values for our simple function v = [x+t, y+t, z+t]
    # Expected: ∂v/∂x = [1,0,0], ∂v/∂y = [0,1,0], ∂v/∂z = [0,0,1], ∂v/∂t = [1,1,1]
    print("\n[2/5] Verifying Jacobian correctness...")
    expected_jac = torch.tensor([
        [1, 0, 0, 1],  # ∂v_x/∂(x,y,z,t)
        [0, 1, 0, 1],  # ∂v_y/∂(x,y,z,t)
        [0, 0, 1, 1],  # ∂v_z/∂(x,y,z,t)
    ], dtype=torch.float32, device=device)

    jac_error = (jacobian[0] - expected_jac).abs().max().item()
    print(f"  Jacobian error: {jac_error:.6f}")
    assert jac_error < 1e-5, "Jacobian values incorrect!"
    print(f"  ✓ Jacobian correctness verified")

    # Test transport equation
    print("\n[3/5] Testing transport equation residual...")
    transport_res = physics.transport_equation_residual(velocity, acceleration, jacobian)
    print(f"  Shape: {transport_res.shape}")
    print(f"  Mean abs value: {transport_res.abs().mean().item():.6f}")
    assert transport_res.shape == (N, 3), "Transport residual shape mismatch!"
    print(f"  ✓ Transport equation test passed")

    # Test divergence
    print("\n[4/5] Testing divergence residual...")
    div_res = physics.divergence_residual(jacobian)
    print(f"  Shape: {div_res.shape}")
    print(f"  Mean abs value: {div_res.abs().mean().item():.6f}")
    # For v = [x+t, y+t, z+t], divergence = 1+1+1 = 3
    expected_div = torch.full((N,), 3.0, device=device)
    div_error = (div_res - expected_div).abs().max().item()
    print(f"  Divergence error: {div_error:.6f}")
    assert div_error < 1e-5, "Divergence values incorrect!"
    assert div_res.shape == (N,), "Divergence residual shape mismatch!"
    print(f"  ✓ Divergence test passed")

    # Test rigid body
    print("\n[5/5] Testing rigid body residual...")
    rigid_res = physics.rigid_body_residual(jacobian)
    print(f"  Shape: {rigid_res.shape}")
    print(f"  Mean abs value: {rigid_res.abs().mean().item():.6f}")
    assert rigid_res.shape == (N,), "Rigid body residual shape mismatch!"
    print(f"  ✓ Rigid body test passed")

    # Test compute_all_residuals
    print("\n[Bonus] Testing compute_all_residuals...")
    residuals = physics.compute_all_residuals(
        test_velocity_func,
        test_acceleration_func,
        xyzt,
        enable_transport=True,
        enable_divergence=True,
        enable_rigid_body=True,
        enable_energy=True
    )
    print(f"  Residuals computed: {list(residuals.keys())}")
    for key, val in residuals.items():
        print(f"    - {key}: shape={val.shape}, mean={val.abs().mean().item():.6f}")
    print(f"  ✓ Batch computation test passed")

    print("\n" + "=" * 60)
    print("✓ All tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_physics_equations()
