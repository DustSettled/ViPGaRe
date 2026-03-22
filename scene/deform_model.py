import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.time_utils import DeformNetwork, CodeField
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func, quaternion_multiply
from utils.velocity_field_utils import VelocityWarpper, SegVel
import einops

# ============ MPM Integration ============
try:
    from mpm_core import MPMConfig, MPMSimulator, MPMPhysicsState
    MPM_AVAILABLE = True
except ImportError:
    print("⚠️  MPM module not found. Physics constraints disabled.")
    print("   To enable: pip install warp-lang")
    MPM_AVAILABLE = False

class DeformModel:
    def __init__(self, is_blender=True, is_6dof=False, max_time=0.7, vel_start_time=0.0, light=True, physics_code=16, use_affine=False,
                 use_mpm=False, mpm_config=None):
        """
        Initialize DeformModel with optional 12-DOF affine transformation and MPM physics.

        Args:
            use_affine: If True, use SegVelAffine (12-DOF) instead of SegVel (6-DOF)
            use_mpm: Enable MPM physics regularization
            mpm_config: Optional MPMConfig
        """
        self.d_gate = False
        self.v_gate = False
        self.use_affine = use_affine
        deform_code_dim = physics_code # 16
        self.code_field = CodeField(D=4, W=128, input_ch=3, output_ch=deform_code_dim, multires=8).cuda()

        # Material Network for learning physics properties (E, nu)
        # Input: deform_code [deform_code_dim] -> Output: [E, nu]
        self.material_net = nn.Sequential(
            nn.Linear(deform_code_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        ).cuda()

        # Initialize material network to output reasonable defaults
        # Last layer bias initialization
        # E = softplus(x) + offset -> want ~1e5
        # nu = sigmoid(x) * 0.49 -> want ~0.3
        nn.init.constant_(self.material_net[-1].weight, 0.0)
        nn.init.constant_(self.material_net[-1].bias, 0.0)
        self.material_net[-1].bias.data[0] = 5.0 # softplus(5) ~ 5, need scaling later
        self.material_net[-1].bias.data[1] = 0.0 # sigmoid(0) = 0.5 -> 0.25

        if light:
            self.deform = DeformNetwork(D=6, W=128, input_ch=3, hyper_ch=deform_code_dim, multires=8,
                                        is_blender=is_blender, is_6dof=is_6dof, gated=self.d_gate).cuda()
        else:
            self.deform = DeformNetwork(D=8, W=256, input_ch=3, hyper_ch=deform_code_dim, multires=8,
                                        is_blender=is_blender, is_6dof=is_6dof, gated=self.d_gate).cuda()

        # Choose velocity field: 6-DOF (SE(3)) or 12-DOF (Affine)
        if use_affine:
            from utils.velocity_field_affine import SegVelAffine
            self.vel_net = SegVelAffine(deform_code_dim=deform_code_dim, hidden_dim=128, layers=5).cuda()
            print("[DeformModel] Using 12-DOF Affine velocity field")
        else:
            self.vel_net = SegVel(deform_code_dim=deform_code_dim, hidden_dim=128, layers=5).cuda()
            print("[DeformModel] Using 6-DOF SE(3) velocity field")

        self.vel = VelocityWarpper(self.vel_net)
        self.optimizer = None
        self.spatial_lr_scale = 5
        self.max_time = max_time
        self.vel_start_time = vel_start_time

        # PINNs-related attributes (will be initialized in train_setting if needed)
        self.pinns_physics_correct = False  # Enable physics correction during inference

        # ============ MPM Physics Integration ============
        self.use_mpm = use_mpm and MPM_AVAILABLE
        self.mpm_simulator = None
        self.mpm_state = None

        if self.use_mpm:
            if mpm_config is None:
                from mpm_core.config import get_default_config
                mpm_config = get_default_config()

            self.mpm_simulator = MPMSimulator(mpm_config)
            print(f"✓ MPM physics enabled in DeformModel")
        else:
            if use_mpm and not MPM_AVAILABLE:
                print("⚠️  MPM requested but not available")

    def step_no_rot(self, xyz, time_emb, deform_code, dt=1/60, max_time=0.75):
        max_time = self.max_time
        min_time = max(self.vel_start_time, dt)
        if time_emb[0, 0] >= min_time and time_emb[0, 0] <= max_time:
            sign = 1 if torch.rand(1)[0] > 0.5 else -1
            deform_time = (time_emb - dt * sign).clamp(self.vel_start_time, max_time)
            gate = self.deform.get_gate(deform_code)
            d_xyz_deform, d_rotation, d_scale = self.deform(xyz.detach(), deform_time, deform_code)
            deform_seg = self.code_field.seg(deform_code)
            xyz_vel = (self.vel.integrate_pos(deform_seg, xyz.detach() + d_xyz_deform.detach(), deform_time, time_emb, dt, rot=False))
            d_xyz_vel = xyz_vel - d_xyz_deform.detach() - xyz.detach()
            d_xyz = d_xyz_vel + d_xyz_deform
            # d_rotation = quaternion_multiply(d_rotation, d_rotation2)
        elif time_emb[0, 0] > max_time:
            # in training, we only consider to integrate once, so we need to modify dt here
            gate = self.deform.get_gate(deform_code)
            deform_time = torch.ones_like(time_emb) * max_time
            dt = time_emb[0, 0] - max_time
            d_xyz_deform, d_rotation, d_scale = self.deform(xyz.detach(), deform_time, deform_code)
            deform_seg = self.code_field.seg(deform_code)
            xyz_vel = (
                self.vel.integrate_pos(deform_seg, xyz.detach() + d_xyz_deform.detach(), deform_time, time_emb, dt, rot=False))
            d_xyz_vel = xyz_vel - d_xyz_deform.detach() - xyz.detach()
            d_xyz = d_xyz_vel + d_xyz_deform
        else:
            gate = self.deform.get_gate(deform_code)
            d_xyz, d_rotation, d_scale = self.deform(xyz.detach(), time_emb, deform_code)
        if self.v_gate:
            d_xyz = d_xyz * gate
            d_rotation = d_rotation * gate
            d_rotation[..., 1:] = d_rotation[..., 1:] + 1
            d_scale = d_scale * gate
        return d_xyz, d_rotation, d_scale

    def step_full_rot(self, xyz, time_emb, deform_code, dt=1/60, max_time=0.75):
        max_time = self.max_time
        if time_emb[0, 0] >= dt and time_emb[0, 0] <= max_time:
            sign = 1 if torch.rand(1)[0] > 0.5 else -1
            deform_time = (time_emb - dt * sign).clamp(0, max_time)
            gate = self.deform.get_gate(deform_code)
            d_xyz_deform, d_rotation, d_scale =  self.deform(xyz.detach(), deform_time, deform_code)
            deform_seg = self.code_field.seg(deform_code)
            xyz_vel, d_rotation2 = (self.vel.integrate_pos(deform_seg, xyz.detach() + d_xyz_deform.detach(), deform_time, time_emb, dt, rot=True))
            d_xyz_vel = xyz_vel - d_xyz_deform.detach() - xyz.detach()
            d_xyz = d_xyz_vel + d_xyz_deform
            d_rotation = quaternion_multiply(d_rotation2, d_rotation)
            # d_rotation = quaternion_multiply(d_rotation, d_rotation2)
        elif time_emb[0, 0] > max_time:
            gate = self.deform.get_gate(deform_code)
            deform_time = torch.ones_like(time_emb) * max_time
            d_xyz_deform, d_rotation, d_scale = self.deform(xyz.detach(), deform_time, deform_code)
            deform_seg = self.code_field.seg(deform_code)
            xyz_vel, d_rotation2 = (
                self.vel.integrate_pos(deform_seg, xyz.detach() + d_xyz_deform.detach(), deform_time, time_emb, dt, rot=True))
            d_xyz_vel = xyz_vel - d_xyz_deform.detach() - xyz.detach()
            d_xyz = d_xyz_vel + d_xyz_deform
            d_rotation = quaternion_multiply(d_rotation2, d_rotation)
            # d_rotation = quaternion_multiply(d_rotation, d_rotation2)
        else:
            gate = self.deform.get_gate(deform_code)
            d_xyz, d_rotation, d_scale = self.deform(xyz.detach(), time_emb, deform_code)
        if self.v_gate:
            d_xyz = d_xyz * gate
            d_rotation = d_rotation * gate
            d_rotation[..., 1:] = d_rotation[..., 1:] + 1
            d_scale = d_scale * gate
        return d_xyz, d_rotation, d_scale

    def step(self, xyz, time_emb, deform_code, dt=1/60, max_time=0.75):
        if xyz.shape[0] == 0:
            return torch.empty((0, 3), device=xyz.device), torch.empty((0, 4), device=xyz.device), torch.empty((0, 3), device=xyz.device)
        max_time = self.max_time
        if time_emb[0, 0] > max_time:
            gate = self.deform.get_gate(deform_code)
            deform_time = torch.ones_like(time_emb) * max_time
            d_xyz_deform, d_rotation, d_scale = self.deform(xyz.detach(), deform_time, deform_code)
            deform_seg = self.code_field.seg(deform_code)
            xyz_vel, d_rotation2 = (
                self.vel.integrate_pos(deform_seg, xyz.detach() + d_xyz_deform.detach(), deform_time, time_emb, dt, rot=True))
            d_xyz_vel = xyz_vel - d_xyz_deform.detach() - xyz.detach()
            d_xyz = d_xyz_vel + d_xyz_deform
            d_rotation = quaternion_multiply(d_rotation2, d_rotation)
            # d_rotation = quaternion_multiply(d_rotation, d_rotation2)
        else:
            gate = self.deform.get_gate(deform_code)
            d_xyz, d_rotation, d_scale = self.deform(xyz.detach(), time_emb, deform_code)
        if self.v_gate:
            d_xyz = d_xyz * gate
            d_rotation = d_rotation * gate
            d_rotation[..., 1:] = d_rotation[..., 1:] + 1
            d_scale = d_scale * gate
        return d_xyz, d_rotation, d_scale

    # def step(self, xyz, time_emb, deform_code, dt=1/60, max_time=0.75):
    #     d_xyz, d_rotation, d_scale = self.deform(deform_code, time_emb)
    #     return d_xyz, d_rotation, d_scale

    # def base_vel_step(self, xyz, time_emb, dt=1/60):
    #     d_xyz = self.vel.integrate_pos(xyz, torch.zeros_like(time_emb), time_emb, dt) - xyz.detach()
    #     return d_xyz

    # def base_vel_step(self, xyz, time_emb, dt=1 / 5):
    #     xyz, d_rot = self.vel.integrate_pos(xyz, torch.zeros_like(time_emb), time_emb, dt, rot=True)
    #     d_xyz = xyz - xyz.detach()
    #     return d_xyz, d_rot

    # def vel_loss(self, deform_code, begin=0., end=1., tmax=0.75, device='cuda'):
    #     t = torch.rand(deform_code.shape[0], 1, device=device) * (end - begin) + begin
    #     x = torch.rand(deform_code.shape[0], 3, device=device) * 2 - 1
    #     # acc = self.deform.get_acc(deform_code, t)
    #     # loss = torch.mean(torch.norm(acc, dim=-1))
    #     acc = self.acc(deform_code.detach(), x)
    #     vel = self.deform.get_local_vel(deform_code.detach(), x, t)
    #     hess = self.deform.get_local_hessian(deform_code.detach(), x.copy(), t.copy())
    #     dvdt = hess[..., -1, -1]
    #     jac_v = hess[..., -1, :-1]
    #     v_jacv = einops.einsum(vel, jac_v, '... xyz, ... uvw xyz -> ... uvw')
    #     lhs = dvdt + v_jacv
    #     # interp
    #     mask = (t < tmax).squeeze(-1)
    #     loss_interp = torch.mean(torch.norm(lhs[mask].detach() - acc[mask], dim=-1))
    #     # extrap
    #     loss_extrap = torch.mean(torch.norm(lhs[~mask] - acc[~mask].detach(), dim=-1))
    #     loss = loss_interp + loss_extrap
    #     return loss

    def train_setting(self, training_args):
        l = [
            {'params': list(self.deform.parameters()) + list(self.vel.parameters()) + list(self.code_field.parameters()) + list(self.material_net.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "deform"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))
        torch.save(self.code_field.state_dict(), os.path.join(out_weights_path, 'code_field.pth'))
        torch.save(self.vel.state_dict(), os.path.join(out_weights_path, 'vel.pth'))
        torch.save(self.material_net.state_dict(), os.path.join(out_weights_path, 'material_net.pth'))

        # 保存 MPM 状态
        if self.use_mpm and self.mpm_state is not None:
            mpm_state_path = os.path.join(out_weights_path, 'mpm_state.pt')
            self.mpm_state.save(mpm_state_path)

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        deform_weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))
        self.deform.load_state_dict(torch.load(deform_weights_path, weights_only=True))
        code_field_weights_path = os.path.join(model_path, "deform/iteration_{}/code_field.pth".format(loaded_iter))
        self.code_field.load_state_dict(torch.load(code_field_weights_path, weights_only=True))
        vel_weights_path = os.path.join(model_path, "deform/iteration_{}/vel.pth".format(loaded_iter))
        self.vel.load_state_dict(torch.load(vel_weights_path, weights_only=True))

        # Load material network if exists
        material_weights_path = os.path.join(model_path, "deform/iteration_{}/material_net.pth".format(loaded_iter))
        if os.path.exists(material_weights_path):
            self.material_net.load_state_dict(torch.load(material_weights_path, weights_only=True))
            print(f"[DeformModel] Loaded material network from {material_weights_path}")

        # 加载 MPM 状态
        if self.use_mpm:
            mpm_state_path = os.path.join(model_path, "deform/iteration_{}/mpm_state.pt".format(loaded_iter))
            if os.path.exists(mpm_state_path):
                from mpm_core import MPMPhysicsState
                self.mpm_state = MPMPhysicsState.load(mpm_state_path)
            else:
                print(f"⚠️  未找到 MPM 状态文件: {mpm_state_path}")

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def apply_physics_correction(self, xyz, d_xyz, d_rotation, time_input, deform_code):
        """
        Apply physics-based correction to deformation predictions during inference.

        This method uses the transport equation to refine predictions in extrapolation regions.

        Args:
            xyz: [N, 3] - original Gaussian positions
            d_xyz: [N, 3] - predicted displacement
            d_rotation: [N, 4] - predicted rotation (quaternion)
            time_input: [N, 1] - time values
            deform_code: [N, K] - deform codes

        Returns:
            corrected_d_xyz: [N, 3] - corrected displacement
            corrected_d_rotation: [N, 4] - corrected rotation
        """
        if not self.pinns_physics_correct or self.training:
            # No correction during training or if disabled
            return d_xyz, d_rotation

        with torch.no_grad():
            # Construct xyzt for physics evaluation
            deformed_xyz = xyz + d_xyz
            xyzt = torch.cat([deformed_xyz, time_input], dim=-1)
            xyzt.requires_grad_(True)

            # Get velocity and acceleration from network
            deform_seg = self.code_field.seg(deform_code)
            velocity = self.vel_net.get_vel(deform_seg, xyzt)
            acceleration = self.vel_net.get_acc(deform_seg, xyzt)

            # Compute small correction based on transport equation residual
            # This is a simple Euler-step correction
            correction_strength = 0.1  # Can be tuned
            d_xyz_corrected = d_xyz + correction_strength * acceleration * time_input

            # Rotation correction (optional, can be disabled)
            d_rotation_corrected = d_rotation

        return d_xyz_corrected, d_rotation_corrected

    # ============ MPM Physics Loss Methods ============

    def get_material_params(self, deform_code):
        """
        Predict material parameters (E, nu) from deform code.

        Args:
            deform_code: [N, K] latent code

        Returns:
            material_params: [N, 2] tensor where:
                - col 0: Young's modulus E > 0
                - col 1: Poisson's ratio 0 <= nu < 0.5
        """
        raw_out = self.material_net(deform_code)

        # E: Softplus ensuring positivity, scaled to reasonable physical range (e.g. 1e4 ~ 1e7)
        # Base value + learned deviation
        log_E = raw_out[..., 0]
        E = F.softplus(log_E) * 1e4 + 1e3

        # nu: Sigmoid to [0, 0.49] (avoid 0.5 instability)
        logit_nu = raw_out[..., 1]
        nu = torch.sigmoid(logit_nu) * 0.49

        return torch.stack([E, nu], dim=-1)

    def init_mpm_state(self, gaussians):
        """
        Initialize MPM physics state from Gaussian model.

        Call this once after loading/creating gaussians.
        """
        if not self.use_mpm:
            return

        from mpm_core import MPMPhysicsState
        self.mpm_state = MPMPhysicsState.from_gaussian_model(
            gaussians=gaussians,
            config=self.mpm_simulator.config,
        )
        print(f"✓ MPM state initialized ({self.mpm_state.n_particles} particles)")

    def compute_mpm_physics_loss(self, xyz, d_xyz, time_emb, velocity=None):
        """
        Compute physics-based loss using MPM simulator.

        Fixes:
        1. F is estimated from velocity field Jacobian (F ≈ I + dt * ∂v/∂x),
           so volume_preservation and elastic_energy losses are non-trivial.
        2. material_params (E, nu) participate in Neo-Hookean elastic energy,
           enabling gradient flow to material_net.

        Args:
            xyz: Canonical Gaussian positions [N, 3]
            d_xyz: Position displacement from deformation network [N, 3]
            time_emb: Time embedding [N, 1]
            velocity: Optional current velocities [N, 3]

        Returns:
            Dictionary of physics loss components
        """
        if not self.use_mpm or self.mpm_state is None:
            return {}

        xyz_deformed = xyz + d_xyz

        # Update state position for boundary check (detach to avoid stale graph)
        self.mpm_state.position = xyz_deformed.detach().clone()
        if velocity is not None:
            self.mpm_state.velocity = velocity.detach().clone()

        # Compute deform code from canonical positions
        deform_code = self.code_field(xyz)
        deform_seg = self.code_field.seg(deform_code)

        # [FIX 1] Estimate deformation gradient F from velocity field Jacobian.
        # F ≈ I + dt * (∂v/∂x) captures instantaneous strain rate.
        # Gradient flows back through vel_net via jac.
        xyzt = torch.cat([xyz_deformed.detach(), time_emb], dim=-1)
        _, jac = self.vel_net.get_vel_jac(deform_seg, xyzt)  # [N, 3, 3]
        dt_mpm = 1.0 / 30.0
        identity = torch.eye(3, device=xyz.device, dtype=xyz.dtype).unsqueeze(0).expand(xyz.shape[0], 3, 3)
        F_computed = identity + dt_mpm * jac  # [N, 3, 3], differentiable through vel_net

        # Update state F for diagnostics/saving (detached copy)
        self.mpm_state.F = F_computed.detach().clone()

        # [FIX 2] Compute material parameters — gradient flows through material_net
        material_params = self.get_material_params(deform_code)
        self.mpm_state.material_params = material_params.detach().clone()

        # Compute Neo-Hookean elastic energy loss (uses both F and material_params)
        return self.mpm_simulator.compute_physics_loss_with_material(
            F=F_computed,
            material_params=material_params,
            position=xyz_deformed,
        )

    def get_mpm_regularization_loss(self, xyz, d_xyz, time_emb, velocity=None, weight=0.1, elastic_weight=0.01):
        """
        Get weighted MPM regularization loss for training.

        Args:
            xyz: Canonical Gaussian positions [N, 3]
            d_xyz: Position displacement from deformation network [N, 3]
            time_emb: Time embedding [N, 1]
            velocity: Optional velocities [N, 3]
            weight: Overall weight for MPM loss
            elastic_weight: Relative weight for elastic_energy term (default 0.01,
                            tunable via --mpm_elastic_weight)

        Returns:
            Scalar loss tensor (differentiable through vel_net and material_net)
        """
        if not self.use_mpm:
            return torch.tensor(0.0, device=xyz.device)

        losses = self.compute_mpm_physics_loss(xyz, d_xyz, time_emb, velocity)

        total_loss = weight * (
            losses.get('elastic_energy', 0.0) * elastic_weight +
            losses.get('volume_preservation', 0.0) +
            losses.get('boundary_penetration', 0.0) * 0.5
        )

        return total_loss

