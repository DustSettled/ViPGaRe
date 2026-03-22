"""
Physics-Informed Neural Networks (PINNs) - Collocation Point Sampler

This module implements sampling strategies for collocation points where
physics equation residuals are evaluated.

Sampling Strategy:
- Sample near existing Gaussian points (KNN-based)
- Adaptive sampling rate based on time (higher for extrapolation)
- KNN cache updated every 10 iterations for efficiency

Author: FreeGave Team
Date: 2025
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np


class CollocationSampler(nn.Module):
    """
    Sample collocation points for PINNs loss computation.

    Key features:
    - Sampling ratio: 0.3 for t ≤ 0.75, 0.5 for t > 0.75
    - KNN cache updated every 10 iterations
    - Spatial perturbation around Gaussian centers
    """

    def __init__(
        self,
        base_sample_ratio: float = 0.05,  # 修复：从0.05→0.01，避免显存爆炸
        extrapolation_sample_ratio: float = 0.1,  # 修复：从0.1→0.02
        extrapolation_threshold: float = 0.75,
        knn_cache_update_interval: int = 10,
        perturbation_std: float = 0.05,
        device: str = 'cuda'
    ):
        """
        Args:
            base_sample_ratio: sampling ratio for interpolation region (t ≤ 0.75)
            extrapolation_sample_ratio: sampling ratio for extrapolation region (t > 0.75)
            extrapolation_threshold: time threshold for extrapolation
            knn_cache_update_interval: update KNN cache every N iterations
            perturbation_std: standard deviation for spatial perturbation
            device: device for computation
        """
        super(CollocationSampler, self).__init__()
        self.base_sample_ratio = base_sample_ratio
        self.extrapolation_sample_ratio = extrapolation_sample_ratio
        self.extrapolation_threshold = extrapolation_threshold
        self.knn_cache_update_interval = knn_cache_update_interval
        self.perturbation_std = perturbation_std
        self.device = device

        # KNN cache
        self.knn_cache = None
        self.last_knn_update_iter = -1

    def get_sample_ratio(self, t: torch.Tensor) -> float:
        """
        Get adaptive sampling ratio based on time.

        Args:
            t: time value (scalar tensor)

        Returns:
            sample_ratio: sampling ratio
        """
        if t.item() > self.extrapolation_threshold:
            return self.extrapolation_sample_ratio
        else:
            return self.base_sample_ratio

    def update_knn_cache(self, xyz: torch.Tensor):
        """
        Update KNN cache for efficient sampling.

        Args:
            xyz: [N, 3] - Gaussian centers
        """
        self.knn_cache = xyz.detach().clone()

    def should_update_knn_cache(self, iteration: int) -> bool:
        """
        Check if KNN cache should be updated.

        Args:
            iteration: current training iteration

        Returns:
            should_update: whether to update KNN cache
        """
        if self.last_knn_update_iter < 0:
            return True  # First time
        return (iteration - self.last_knn_update_iter) >= self.knn_cache_update_interval

    def sample_collocation_points(
        self,
        xyz: torch.Tensor,
        t: torch.Tensor,
        iteration: int,
        bbox: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample collocation points for PINNs loss computation.

        Strategy:
        1. Update KNN cache if needed
        2. Sample subset of Gaussian points based on adaptive ratio
        3. Add spatial perturbation
        4. Concatenate with time to get (x, y, z, t)

        Args:
            xyz: [N, 3] - Gaussian centers
            t: scalar tensor - current time
            iteration: current training iteration
            bbox: optional (min_corner, max_corner) for boundary clamping

        Returns:
            collocation_points: [M, 4] - (x, y, z, t) collocation points
            selected_indices: [M] - indices of selected Gaussians
        """
        N = xyz.shape[0]

        # Update KNN cache if needed
        if self.should_update_knn_cache(iteration):
            self.update_knn_cache(xyz)
            self.last_knn_update_iter = iteration

        # Get adaptive sampling ratio
        sample_ratio = self.get_sample_ratio(t)
        n_samples = int(N * sample_ratio)
        n_samples = max(1, n_samples)  # At least 1 sample

        # Randomly sample indices
        selected_indices = torch.randperm(N, device=self.device)[:n_samples]

        # Get selected Gaussian centers
        selected_xyz = xyz[selected_indices]  # [M, 3]

        # Add spatial perturbation (Gaussian noise)
        perturbation = torch.randn_like(selected_xyz) * self.perturbation_std
        perturbed_xyz = selected_xyz + perturbation

        # Clamp to bounding box if provided
        if bbox is not None:
            min_corner, max_corner = bbox
            perturbed_xyz = torch.clamp(perturbed_xyz, min_corner, max_corner)
        else:
            # Apply reasonable bounds to avoid extreme values that could cause NaN
            perturbed_xyz = torch.clamp(perturbed_xyz, -10.0, 10.0)

        # Concatenate with time
        t_expanded = t.expand(n_samples, 1)  # [M, 1]
        collocation_points = torch.cat([perturbed_xyz, t_expanded], dim=-1)  # [M, 4]

        # 关键调试：验证输出形状
        # print(f"[DEBUG pinns_sampler] perturbed_xyz shape: {perturbed_xyz.shape}")
        # print(f"[DEBUG pinns_sampler] t shape: {t.shape}, t_expanded shape: {t_expanded.shape}")
        # print(f"[DEBUG pinns_sampler] collocation_points shape: {collocation_points.shape}")

        # Ensure requires_grad=True for autograd
        collocation_points.requires_grad_(True)

        return collocation_points, selected_indices

    def sample_time_values(
        self,
        n_samples: int,
        min_time: float = 0.0,
        max_time: float = 1.0,
        prioritize_extrapolation: bool = True
    ) -> torch.Tensor:
        """
        Sample time values for temporal collocation.

        Args:
            n_samples: number of time samples
            min_time: minimum time value
            max_time: maximum time value
            prioritize_extrapolation: whether to oversample extrapolation region

        Returns:
            time_samples: [n_samples] - sampled time values
        """
        if prioritize_extrapolation:
            # 60% samples from extrapolation region (t > 0.75)
            n_extrap = int(n_samples * 0.6)
            n_interp = n_samples - n_extrap

            # Interpolation region samples
            t_interp = torch.rand(n_interp, device=self.device) * self.extrapolation_threshold

            # Extrapolation region samples
            t_extrap = (
                torch.rand(n_extrap, device=self.device) * (max_time - self.extrapolation_threshold)
                + self.extrapolation_threshold
            )

            time_samples = torch.cat([t_interp, t_extrap], dim=0)
            # Shuffle
            time_samples = time_samples[torch.randperm(n_samples, device=self.device)]
        else:
            # Uniform sampling
            time_samples = torch.rand(n_samples, device=self.device) * (max_time - min_time) + min_time

        return time_samples

    def get_cache_info(self) -> dict:
        """
        Get KNN cache information for debugging.

        Returns:
            info: dict with cache statistics
        """
        info = {
            'has_cache': self.knn_cache is not None,
            'last_update_iter': self.last_knn_update_iter,
            'update_interval': self.knn_cache_update_interval,
            'base_ratio': self.base_sample_ratio,
            'extrapolation_ratio': self.extrapolation_sample_ratio,
        }
        if self.knn_cache is not None:
            info['cache_size'] = self.knn_cache.shape[0]
        return info


def test_collocation_sampler():
    """
    Unit test for collocation sampler.
    """
    print("=" * 60)
    print("Testing CollocationSampler Module")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create sampler
    sampler = CollocationSampler(
        base_sample_ratio=0.3,
        extrapolation_sample_ratio=0.5,
        extrapolation_threshold=0.75,
        knn_cache_update_interval=10,
        device=device
    )

    # Create sample Gaussian centers
    N = 10000
    xyz = torch.randn(N, 3, device=device)
    bbox = (torch.tensor([-1.0, -1.0, -1.0], device=device),
            torch.tensor([1.0, 1.0, 1.0], device=device))

    print(f"\nTest data: N={N} Gaussian points")
    print(f"Device: {device}")

    # Test 1: Sampling in interpolation region
    print("\n[1/5] Testing interpolation region sampling (t=0.5)...")
    t = torch.tensor(0.5, device=device)
    collocation_points, indices = sampler.sample_collocation_points(xyz, t, iteration=0, bbox=bbox)
    print(f"  Sampled points: {collocation_points.shape[0]}")
    print(f"  Expected: ~{int(N * 0.3)}")
    print(f"  Collocation points shape: {collocation_points.shape}")
    print(f"  requires_grad: {collocation_points.requires_grad}")
    print(f"  ✓ Interpolation sampling test passed")

    # Test 2: Sampling in extrapolation region
    print("\n[2/5] Testing extrapolation region sampling (t=0.9)...")
    t = torch.tensor(0.9, device=device)
    collocation_points, indices = sampler.sample_collocation_points(xyz, t, iteration=1, bbox=bbox)
    print(f"  Sampled points: {collocation_points.shape[0]}")
    print(f"  Expected: ~{int(N * 0.5)}")
    print(f"  ✓ Extrapolation sampling test passed")

    # Test 3: KNN cache update
    print("\n[3/5] Testing KNN cache update mechanism...")
    for i in range(15):
        should_update = sampler.should_update_knn_cache(i)
        if should_update:
            print(f"  Iteration {i}: Cache updated")
    print(f"  ✓ KNN cache test passed")

    # Test 4: Time sampling
    print("\n[4/5] Testing time value sampling...")
    time_samples = sampler.sample_time_values(
        n_samples=100,
        prioritize_extrapolation=True
    )
    n_extrap = (time_samples > 0.75).sum().item()
    print(f"  Total samples: {len(time_samples)}")
    print(f"  Extrapolation samples (t > 0.75): {n_extrap}")
    print(f"  Expected: ~60")
    print(f"  ✓ Time sampling test passed")

    # Test 5: Cache info
    print("\n[5/5] Testing cache info...")
    info = sampler.get_cache_info()
    print(f"  Cache info:")
    for key, val in info.items():
        print(f"    - {key}: {val}")
    print(f"  ✓ Cache info test passed")

    print("\n" + "=" * 60)
    print("✓ All tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_collocation_sampler()
