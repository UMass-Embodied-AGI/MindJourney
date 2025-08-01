from typing import Literal

import numpy as np
import roma
import scipy.interpolate
import torch
import torch.nn.functional as F

DEFAULT_FOV_RAD = 0.9424777960769379  # 54 degrees by default


def get_camera_dist(
    source_c2ws: torch.Tensor,  # N x 3 x 4
    target_c2ws: torch.Tensor,  # M x 3 x 4
    mode: str = "translation",
):
    if mode == "rotation":
        dists = torch.acos(
            (
                (
                    torch.matmul(
                        source_c2ws[:, None, :3, :3],
                        target_c2ws[None, :, :3, :3].transpose(-1, -2),
                    )
                    .diagonal(offset=0, dim1=-2, dim2=-1)
                    .sum(-1)
                    - 1
                )
                / 2
            ).clamp(-1, 1)
        ) * (180 / torch.pi)
    elif mode == "translation":
        dists = torch.norm(
            source_c2ws[:, None, :3, 3] - target_c2ws[None, :, :3, 3], dim=-1
        )
    else:
        raise NotImplementedError(
            f"Mode {mode} is not implemented for finding nearest source indices."
        )
    return dists


def to_hom(X):
    # get homogeneous coordinates of the input
    X_hom = torch.cat([X, torch.ones_like(X[..., :1])], dim=-1)
    return X_hom


def to_hom_pose(pose):
    # get homogeneous coordinates of the input pose
    if pose.shape[-2:] == (3, 4):
        pose_hom = torch.eye(4, device=pose.device)[None].repeat(pose.shape[0], 1, 1)
        pose_hom[:, :3, :] = pose
        return pose_hom
    return pose


def get_default_intrinsics(
    fov_rad=DEFAULT_FOV_RAD,
    aspect_ratio=1.0,
):
    if not isinstance(fov_rad, torch.Tensor):
        fov_rad = torch.tensor(
            [fov_rad] if isinstance(fov_rad, (int, float)) else fov_rad
        )
    if aspect_ratio >= 1.0:  # W >= H
        focal_x = 0.5 / torch.tan(0.5 * fov_rad)
        focal_y = focal_x * aspect_ratio
    else:  # W < H
        focal_y = 0.5 / torch.tan(0.5 * fov_rad)
        focal_x = focal_y / aspect_ratio
    intrinsics = focal_x.new_zeros((focal_x.shape[0], 3, 3))
    intrinsics[:, torch.eye(3, device=focal_x.device, dtype=bool)] = torch.stack(
        [focal_x, focal_y, torch.ones_like(focal_x)], dim=-1
    )
    intrinsics[:, :, -1] = torch.tensor(
        [0.5, 0.5, 1.0], device=focal_x.device, dtype=focal_x.dtype
    )
    return intrinsics


def get_image_grid(img_h, img_w):
    # add 0.5 is VERY important especially when your img_h and img_w
    # is not very large (e.g., 72)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    y_range = torch.arange(img_h, dtype=torch.float32).add_(0.5)
    x_range = torch.arange(img_w, dtype=torch.float32).add_(0.5)
    Y, X = torch.meshgrid(y_range, x_range, indexing="ij")  # [H,W]
    xy_grid = torch.stack([X, Y], dim=-1).view(-1, 2)  # [HW,2]
    return to_hom(xy_grid)  # [HW,3]


def img2cam(X, cam_intr):
    return X @ cam_intr.inverse().transpose(-1, -2)


def cam2world(X, pose):
    X_hom = to_hom(X)
    pose_inv = torch.linalg.inv(to_hom_pose(pose))[..., :3, :4]
    return X_hom @ pose_inv.transpose(-1, -2)


def get_center_and_ray(
    img_h, img_w, pose, intr, zero_center_for_debugging=False
):  # [HW,2]
    # given the intrinsic/extrinsic matrices, get the camera center and ray directions]
    # assert(opt.camera.model=="perspective")

    # compute center and ray
    grid_img = get_image_grid(img_h, img_w)  # [HW,3]
    grid_3D_cam = img2cam(grid_img.to(intr.device), intr.float())  # [B,HW,3]
    center_3D_cam = torch.zeros_like(grid_3D_cam)  # [B,HW,3]

    # transform from camera to world coordinates
    grid_3D = cam2world(grid_3D_cam, pose)  # [B,HW,3]
    center_3D = cam2world(center_3D_cam, pose)  # [B,HW,3]
    ray = grid_3D - center_3D  # [B,HW,3]

    return center_3D_cam if zero_center_for_debugging else center_3D, ray, grid_3D_cam


def get_plucker_coordinates(
    extrinsics_src,
    extrinsics,
    intrinsics=None,
    fov_rad=DEFAULT_FOV_RAD,
    mode="plucker",
    rel_zero_translation=True,
    zero_center_for_debugging=False,
    target_size=[72, 72],  # 576-size image
    return_grid_cam=False,  # save for later use if want restore
):
    if intrinsics is None:
        intrinsics = get_default_intrinsics(fov_rad).to(extrinsics.device)
    else:
        # for some data preprocessed in the early stage (e.g., MVI and CO3D),
        # intrinsics are expressed in raw pixel space (e.g., 576x576) instead
        # of normalized image coordinates
        if not (
            torch.all(intrinsics[:, :2, -1] >= 0)
            and torch.all(intrinsics[:, :2, -1] <= 1)
        ):
            intrinsics[:, :2] /= intrinsics.new_tensor(target_size).view(1, -1, 1) * 8
        # you should ensure the intrisics are expressed in
        # resolution-independent normalized image coordinates just performing a
        # very simple verification here checking if principal points are
        # between 0 and 1
        assert (
            torch.all(intrinsics[:, :2, -1] >= 0)
            and torch.all(intrinsics[:, :2, -1] <= 1)
        ), "Intrinsics should be expressed in resolution-independent normalized image coordinates."

    c2w_src = torch.linalg.inv(extrinsics_src)
    if not rel_zero_translation:
        c2w_src[:3, 3] = c2w_src[3, :3] = 0.0
    # transform coordinates from the source camera's coordinate system to the coordinate system of the respective camera
    extrinsics_rel = torch.einsum(
        "vnm,vmp->vnp", extrinsics, c2w_src[None].repeat(extrinsics.shape[0], 1, 1)
    )

    intrinsics[:, :2] *= extrinsics.new_tensor(
        [
            target_size[1],  # w
            target_size[0],  # h
        ]
    ).view(1, -1, 1)
    centers, rays, grid_cam = get_center_and_ray(
        img_h=target_size[0],
        img_w=target_size[1],
        pose=extrinsics_rel[:, :3, :],
        intr=intrinsics,
        zero_center_for_debugging=zero_center_for_debugging,
    )

    if mode == "plucker" or "v1" in mode:
        rays = torch.nn.functional.normalize(rays, dim=-1)
        plucker = torch.cat((rays, torch.cross(centers, rays, dim=-1)), dim=-1)
    else:
        raise ValueError(f"Unknown Plucker coordinate mode: {mode}")

    plucker = plucker.permute(0, 2, 1).reshape(plucker.shape[0], -1, *target_size)
    if return_grid_cam:
        return plucker, grid_cam.reshape(-1, *target_size, 3)
    return plucker


def rt_to_mat4(
    R: torch.Tensor, t: torch.Tensor, s: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Args:
        R (torch.Tensor): (..., 3, 3).
        t (torch.Tensor): (..., 3).
        s (torch.Tensor): (...,).

    Returns:
        torch.Tensor: (..., 4, 4)
    """
    mat34 = torch.cat([R, t[..., None]], dim=-1)
    if s is None:
        bottom = (
            mat34.new_tensor([[0.0, 0.0, 0.0, 1.0]])
            .reshape((1,) * (mat34.dim() - 2) + (1, 4))
            .expand(mat34.shape[:-2] + (1, 4))
        )
    else:
        bottom = F.pad(1.0 / s[..., None, None], (3, 0), value=0.0)
    mat4 = torch.cat([mat34, bottom], dim=-2)
    return mat4

import roma

def get_inplace_yaw_w2cs(
    ref_w2c: torch.Tensor,
    angle_deg: float,
    num_frames: int,
    turn_direction: Literal["left","right"]="right",
    endpoint: bool = False,
) -> torch.Tensor:
    """
    Generate a sequence of w2c matrices by rotating the camera in-place
    around its own 'up' axis by +/- angle_deg.  Position stays fixed.

    Args:
        ref_w2c:       (4,4) reference world-to-camera matrix.
        angle_deg:     How many degrees to rotate in total.
        num_frames:    Number of output poses.
        turn_direction: "right" => negative sign, "left" => positive sign.
        endpoint:      Whether to include the final angle or not in the interpolation.

    Returns:
        w2cs: (num_frames, 4, 4) new world-to-camera matrices.
    """
    device = ref_w2c.device
    # Invert reference to get c2w.
    ref_c2w = torch.linalg.inv(ref_w2c)

    # Camera's local "up" axis. In many OpenCV conventions, c2w[:3,1] is downward,
    # so negative of it is the 'up' direction. You may adjust if your convention differs.
    # up_c = -ref_c2w[:3,1]
    up_c = torch.tensor([0.0, 1.0, 0.0], device=device)

    # Decide sign for left vs right turn.
    sign = 1.0 if turn_direction == "left" else - 1.0
    total_angle_rad = sign * (angle_deg * (np.pi / 180.0))

    # Interpolate angles from 0..(± total_angle_rad)
    if endpoint:
        thetas = torch.linspace(0.0, total_angle_rad, num_frames, device=device)
    else:
        # Just like the other motion functions: generate (num_frames+1) then drop the last
        # so that the final angle is not repeated.
        thetas = torch.linspace(0.0, total_angle_rad, num_frames+1, device=device)[:-1]

    w2cs = []
    for theta in thetas:
        # Rotation around camera's up axis by 'theta'
        R = roma.rotvec_to_rotmat(up_c * theta)
        # Keep the same position, but update orientation.
        c2w_i = ref_c2w.clone()
        c2w_i[:3,:3] = ref_c2w[:3,:3] @ R

        # Convert back to w2c
        w2c_i = torch.linalg.inv(c2w_i)
        w2cs.append(w2c_i.unsqueeze(0))

    return torch.cat(w2cs, dim=0)  # (num_frames, 4,4)

def get_preset_pose_fov(
    option: Literal[
        "orbit",
        "spiral",
        "lemniscate",
        "zoom-in",
        "zoom-out",
        "dolly zoom-in",
        "dolly zoom-out",
        "move-forward",
        "move-backward",
        "move-up",
        "move-down",
        "move-left",
        "move-right",
        "roll",
        "move-forward-0.25m",
        "move-backward-0.25m"
    ],
    num_frames: int,
    start_w2c: torch.Tensor,
    look_at: torch.Tensor,
    up_direction: torch.Tensor | None = None,
    fov: float = DEFAULT_FOV_RAD,
    spiral_radii: list[float] = [0.5, 0.5, 0.2],
    zoom_factor: float | None = None,
):
    poses = fovs = None
    if option == "orbit":
        poses = torch.linalg.inv(
            get_arc_horizontal_w2cs(
                start_w2c,
                look_at,
                up_direction,
                num_frames=num_frames,
                endpoint=False,
            )
        ).numpy()
        fovs = np.full((num_frames,), fov)
    elif option == "spiral":
        poses = generate_spiral_path(
            torch.linalg.inv(start_w2c)[None].numpy() @ np.diagflat([1, -1, -1, 1]),
            np.array([1, 5]),
            n_frames=num_frames,
            n_rots=2,
            zrate=0.5,
            radii=spiral_radii,
            endpoint=False,
        ) @ np.diagflat([1, -1, -1, 1])
        poses = np.concatenate(
            [
                poses,
                np.array([0.0, 0.0, 0.0, 1.0])[None, None].repeat(len(poses), 0),
            ],
            1,
        )
        # We want the spiral trajectory to always start from start_w2c. Thus we
        # apply the relative pose to get the final trajectory.
        poses = (
            np.linalg.inv(start_w2c.numpy())[None] @ np.linalg.inv(poses[:1]) @ poses
        )
        fovs = np.full((num_frames,), fov)
    elif option == "lemniscate":
        poses = torch.linalg.inv(
            get_lemniscate_w2cs(
                start_w2c,
                look_at,
                up_direction,
                num_frames,
                degree=60.0,
                endpoint=False,
            )
        ).numpy()
        fovs = np.full((num_frames,), fov)
    elif option == "roll":
        poses = torch.linalg.inv(
            get_roll_w2cs(
                start_w2c,
                look_at,
                None,
                num_frames,
                degree=360.0,
                endpoint=False,
            )
        ).numpy()
        fovs = np.full((num_frames,), fov)
    elif option in [
        "dolly zoom-in",
        "dolly zoom-out",
        "zoom-in",
        "zoom-out",
    ]:
        if option.startswith("dolly"):
            direction = "backward" if option == "dolly zoom-in" else "forward"
            poses = torch.linalg.inv(
                get_moving_w2cs(
                    start_w2c,
                    look_at,
                    up_direction,
                    num_frames,
                    endpoint=True,
                    direction=direction,
                )
            ).numpy()
        else:
            poses = torch.linalg.inv(start_w2c)[None].repeat(num_frames, 1, 1).numpy()
        fov_rad_start = fov
        if zoom_factor is None:
            zoom_factor = 0.28 if option.endswith("zoom-in") else 1.5
        fov_rad_end = zoom_factor * fov
        fovs = (
            np.linspace(0, 1, num_frames) * (fov_rad_end - fov_rad_start)
            + fov_rad_start
        )
    elif option in [
        "move-forward",
        "move-backward",
        "move-up",
        "move-down",
        "move-left",
        "move-right",
    ]:
        poses = torch.linalg.inv(
            get_moving_w2cs(
                start_w2c,
                look_at,
                up_direction,
                num_frames,
                endpoint=True,
                direction=option.removeprefix("move-"),
            )
        ).numpy()
        fovs = np.full((num_frames,), fov)
    elif option.startswith("turn-right"):
        angle_deg = float(option.removeprefix("turn-right-"))
        w2cs = get_inplace_yaw_w2cs(
            ref_w2c=start_w2c,
            angle_deg=angle_deg,
            num_frames=num_frames,
            turn_direction="right",
            endpoint=False,
        )
        poses = w2cs.cpu().numpy()
        fovs = np.full((num_frames,), fov)
    elif option.startswith("turn-left"):
        angle_deg = float(option.removeprefix("turn-left-"))
        w2cs = get_inplace_yaw_w2cs(
            ref_w2c=start_w2c,
            angle_deg=30.0,
            num_frames=num_frames,
            turn_direction="left",
            endpoint=False,
        )
        poses = w2cs.cpu().numpy()
        fovs = np.full((num_frames,), fov)
    elif option.startswith("move-forward"):
        dist_meter = float(option.removeprefix("move-forward-"))
        w2cs = get_moving_w2cs(
            ref_w2c=start_w2c,
            lookat=look_at,
            up=up_direction,
            num_frames=num_frames,
            endpoint=True,      # Move a full 0.25 at the final frame
            direction="backward",
            move_distance=dist_meter/2.5, # <--- specify 0.25 m
        )
        poses = w2cs.cpu().numpy()
        fovs = np.full((num_frames,), fov)
    elif option.startswith("move-backward"):
        dist_meter = float(option.removeprefix("move-forward-"))
        w2cs = get_moving_w2cs(
            ref_w2c=start_w2c,
            lookat=look_at,
            up=up_direction,
            num_frames=num_frames,
            endpoint=True,      # Move a full 0.25 at the final frame
            direction="forward",
            move_distance=dist_meter/2.5, # <--- specify 0.25 m
        )
        poses = w2cs.cpu().numpy()
        fovs = np.full((num_frames,), fov)
    else:
        raise ValueError(f"Unknown preset option {option}.")

    return poses, fovs


def get_lookat(origins: torch.Tensor, viewdirs: torch.Tensor) -> torch.Tensor:
    """Triangulate a set of rays to find a single lookat point.

    Args:
        origins (torch.Tensor): A (N, 3) array of ray origins.
        viewdirs (torch.Tensor): A (N, 3) array of ray view directions.

    Returns:
        torch.Tensor: A (3,) lookat point.
    """

    viewdirs = torch.nn.functional.normalize(viewdirs, dim=-1)
    eye = torch.eye(3, device=origins.device, dtype=origins.dtype)[None]
    # Calculate projection matrix I - rr^T
    I_min_cov = eye - (viewdirs[..., None] * viewdirs[..., None, :])
    # Compute sum of projections
    sum_proj = I_min_cov.matmul(origins[..., None]).sum(dim=-3)
    # Solve for the intersection point using least squares
    lookat = torch.linalg.lstsq(I_min_cov.sum(dim=-3), sum_proj).solution[..., 0]
    # Check NaNs.
    assert not torch.any(torch.isnan(lookat))
    return lookat


def get_lookat_w2cs(
    positions: torch.Tensor,
    lookat: torch.Tensor,
    up: torch.Tensor,
    face_off: bool = False,
):
    """
    Args:
        positions: (N, 3) tensor of camera positions
        lookat: (3,) tensor of lookat point
        up: (3,) or (N, 3) tensor of up vector

    Returns:
        w2cs: (N, 3, 3) tensor of world to camera rotation matrices
    """
    forward_vectors = F.normalize(lookat - positions, dim=-1)
    if face_off:
        forward_vectors = -forward_vectors
    if up.dim() == 1:
        up = up[None]
    right_vectors = F.normalize(torch.cross(forward_vectors, up, dim=-1), dim=-1)
    down_vectors = F.normalize(
        torch.cross(forward_vectors, right_vectors, dim=-1), dim=-1
    )
    Rs = torch.stack([right_vectors, down_vectors, forward_vectors], dim=-1)
    w2cs = torch.linalg.inv(rt_to_mat4(Rs, positions))
    return w2cs


def get_arc_horizontal_w2cs(
    ref_w2c: torch.Tensor,
    lookat: torch.Tensor,
    up: torch.Tensor | None,
    num_frames: int,
    clockwise: bool = True,
    face_off: bool = False,
    endpoint: bool = False,
    degree: float = 360.0,
    ref_up_shift: float = 0.0,
    ref_radius_scale: float = 1.0,
    **_,
) -> torch.Tensor:
    ref_c2w = torch.linalg.inv(ref_w2c)
    ref_position = ref_c2w[:3, 3]
    if up is None:
        up = -ref_c2w[:3, 1]
    assert up is not None
    ref_position += up * ref_up_shift
    ref_position *= ref_radius_scale
    thetas = (
        torch.linspace(0.0, torch.pi * degree / 180, num_frames, device=ref_w2c.device)
        if endpoint
        else torch.linspace(
            0.0, torch.pi * degree / 180, num_frames + 1, device=ref_w2c.device
        )[:-1]
    )
    if not clockwise:
        thetas = -thetas
    positions = (
        torch.einsum(
            "nij,j->ni",
            roma.rotvec_to_rotmat(thetas[:, None] * up[None]),
            ref_position - lookat,
        )
        + lookat
    )
    return get_lookat_w2cs(positions, lookat, up, face_off=face_off)


def get_lemniscate_w2cs(
    ref_w2c: torch.Tensor,
    lookat: torch.Tensor,
    up: torch.Tensor | None,
    num_frames: int,
    degree: float,
    endpoint: bool = False,
    **_,
) -> torch.Tensor:
    ref_c2w = torch.linalg.inv(ref_w2c)
    a = torch.linalg.norm(ref_c2w[:3, 3] - lookat) * np.tan(degree / 360 * np.pi)
    # Lemniscate curve in camera space. Starting at the origin.
    thetas = (
        torch.linspace(0, 2 * torch.pi, num_frames, device=ref_w2c.device)
        if endpoint
        else torch.linspace(0, 2 * torch.pi, num_frames + 1, device=ref_w2c.device)[:-1]
    ) + torch.pi / 2
    positions = torch.stack(
        [
            a * torch.cos(thetas) / (1 + torch.sin(thetas) ** 2),
            a * torch.cos(thetas) * torch.sin(thetas) / (1 + torch.sin(thetas) ** 2),
            torch.zeros(num_frames, device=ref_w2c.device),
        ],
        dim=-1,
    )
    # Transform to world space.
    positions = torch.einsum(
        "ij,nj->ni", ref_c2w[:3], F.pad(positions, (0, 1), value=1.0)
    )
    if up is None:
        up = -ref_c2w[:3, 1]
    assert up is not None
    return get_lookat_w2cs(positions, lookat, up)


def get_moving_w2cs(
    ref_w2c: torch.Tensor,
    lookat: torch.Tensor,
    up: torch.Tensor | None,
    num_frames: int,
    endpoint: bool = False,
    direction: str = "forward",
    tilt_xy: torch.Tensor = None,
    move_distance: float = 1.0,  # <--- new parameter
):
    """
    Generate (num_frames) new world-to-camera poses by translating
    the camera from its reference position in a chosen direction,
    for a total distance 'move_distance'.

    Args:
        ref_w2c:       (4,4) reference world-to-camera.
        lookat:        (3,) point for direction reference.
        up:            (3,) global or reference up vector (if None, we infer it).
        num_frames:    number of frames in the output.
        endpoint:      if True, the final pose is the full distance away.
        direction:     one of [forward, backward, up, down, left, right].
        tilt_xy:       optional additional (x,y) tilt offset in the final position.
        move_distance: total distance in *scene units* to move from start to end.

    Returns:
        w2cs: (num_frames, 4, 4) each frame's world-to-camera pose.
    """
    ref_c2w = torch.linalg.inv(ref_w2c)   # (4,4)
    ref_position = ref_c2w[:3, 3]        # (3,)

    # If user didn't supply 'up', infer from c2w (note: c2w[:3,1] points 'down' in OpenCV).
    if up is None:
        up = -ref_c2w[:3, 1]  # negative of c2w[:3,1]

    # Our canonical direction vectors based on the reference camera orientation
    direction_vectors = {
        "forward":  -(lookat - ref_position).clone(),
        "backward": (lookat - ref_position).clone(),
        "up":        up.clone(),
        "down":     -up.clone(),
        "right":     torch.cross((lookat - ref_position), up, dim=0),
        "left":     -torch.cross((lookat - ref_position), up, dim=0),
    }
    if direction not in direction_vectors:
        raise ValueError(f"Invalid direction: {direction}.")

    # Our direction vector, normalized, times 'move_distance'
    dir_vec = F.normalize(direction_vectors[direction], dim=0)
    dir_vec = dir_vec * move_distance   # total displacement we want

    # We'll interpolate from 0 to 1, then multiply by dir_vec.  If endpoint=True,
    # the final pose is exactly the entire shift.  If endpoint=False, we typically
    # do (num_frames+1) then drop the last to avoid duplication with next segment.
    if endpoint:
        alphas = torch.linspace(0, 1, num_frames, device=ref_w2c.device)  # includes 1
    else:
        alphas = torch.linspace(0, 1, num_frames + 1, device=ref_w2c.device)[:-1]

    # Accumulate the final positions
    positions = ref_position + alphas[:, None] * dir_vec[None, :]

    # Optionally add the tilt
    if tilt_xy is not None:
        positions[:, :2] += tilt_xy

    # For each position, build a w2c that *looks* from that position toward the same lookat
    # (with a consistent up).
    w2cs = []
    for pos in positions:
        # Build c2w
        forward = F.normalize(lookat - pos, dim=0)
        right   = F.normalize(torch.cross(forward, up, dim=0), dim=0)
        true_up = F.normalize(torch.cross(right, forward, dim=0), dim=0)

        # c2w rotation = [right, -true_up, forward], consistent with typical OpenCV usage
        # (where +y in camera coords is 'down' if forward is +z).
        R = torch.stack([right, -true_up, forward], dim=1)  # 3x3
        c2w_i = torch.eye(4, device=ref_w2c.device)
        c2w_i[:3, :3] = R
        c2w_i[:3, 3]  = pos

        # invert => w2c
        w2cs.append(torch.linalg.inv(c2w_i))

    return torch.stack(w2cs, dim=0)  # (num_frames,4,4)


def get_roll_w2cs(
    ref_w2c: torch.Tensor,
    lookat: torch.Tensor,
    up: torch.Tensor | None,
    num_frames: int,
    endpoint: bool = False,
    degree: float = 360.0,
    **_,
) -> torch.Tensor:
    ref_c2w = torch.linalg.inv(ref_w2c)
    ref_position = ref_c2w[:3, 3]
    if up is None:
        up = -ref_c2w[:3, 1]  # Infer the up vector from the reference.

    # Create vertical angles
    thetas = (
        torch.linspace(0.0, torch.pi * degree / 180, num_frames, device=ref_w2c.device)
        if endpoint
        else torch.linspace(
            0.0, torch.pi * degree / 180, num_frames + 1, device=ref_w2c.device
        )[:-1]
    )[:, None]

    lookat_vector = F.normalize(lookat[None].float(), dim=-1)
    up = up[None]
    up = (
        up * torch.cos(thetas)
        + torch.cross(lookat_vector, up) * torch.sin(thetas)
        + lookat_vector
        * torch.einsum("ij,ij->i", lookat_vector, up)[:, None]
        * (1 - torch.cos(thetas))
    )

    # Normalize the camera orientation
    return get_lookat_w2cs(ref_position[None].repeat(num_frames, 1), lookat, up)


def normalize(x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)


def viewmatrix(lookdir, up, position, subtract_position=False):
    """Construct lookat view matrix."""
    vec2 = normalize((lookdir - position) if subtract_position else lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def poses_avg(poses):
    """New pose using average position, z-axis, and up vector of input poses."""
    position = poses[:, :3, 3].mean(0)
    z_axis = poses[:, :3, 2].mean(0)
    up = poses[:, :3, 1].mean(0)
    cam2world = viewmatrix(z_axis, up, position)
    return cam2world


def generate_spiral_path(
    poses, bounds, n_frames=120, n_rots=2, zrate=0.5, endpoint=False, radii=None
):
    """Calculates a forward facing spiral path for rendering."""
    # Find a reasonable 'focus depth' for this dataset as a weighted average
    # of near and far bounds in disparity space.
    close_depth, inf_depth = bounds.min() * 0.9, bounds.max() * 5.0
    dt = 0.75
    focal = 1 / ((1 - dt) / close_depth + dt / inf_depth)

    # Get radii for spiral path using 90th percentile of camera positions.
    positions = poses[:, :3, 3]
    if radii is None:
        radii = np.percentile(np.abs(positions), 90, 0)
    radii = np.concatenate([radii, [1.0]])

    # Generate poses for spiral path.
    render_poses = []
    cam2world = poses_avg(poses)
    up = poses[:, :3, 1].mean(0)
    for theta in np.linspace(0.0, 2.0 * np.pi * n_rots, n_frames, endpoint=endpoint):
        t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0]
        position = cam2world @ t
        lookat = cam2world @ [0, 0, -focal, 1.0]
        z_axis = position - lookat
        render_poses.append(viewmatrix(z_axis, up, position))
    render_poses = np.stack(render_poses, axis=0)
    return render_poses


def generate_interpolated_path(
    poses: np.ndarray,
    n_interp: int,
    spline_degree: int = 5,
    smoothness: float = 0.03,
    rot_weight: float = 0.1,
    endpoint: bool = False,
):
    """Creates a smooth spline path between input keyframe camera poses.

    Spline is calculated with poses in format (position, lookat-point, up-point).

    Args:
      poses: (n, 3, 4) array of input pose keyframes.
      n_interp: returned path will have n_interp * (n - 1) total poses.
      spline_degree: polynomial degree of B-spline.
      smoothness: parameter for spline smoothing, 0 forces exact interpolation.
      rot_weight: relative weighting of rotation/translation in spline solve.

    Returns:
      Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
    """

    def poses_to_points(poses, dist):
        """Converts from pose matrices to (position, lookat, up) format."""
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(points):
        """Converts from (position, lookat, up) format to pose matrices."""
        return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])

    def interp(points, n, k, s):
        """Runs multidimensional B-spline interpolation on the input points."""
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=endpoint)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
        return new_points

    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(
        points, n_interp * (points.shape[0] - 1), k=spline_degree, s=smoothness
    )
    return points_to_poses(new_points)


def similarity_from_cameras(c2w, strict_scaling=False, center_method="focus"):
    """
    reference: nerf-factory
    Get a similarity transform to normalize dataset
    from c2w (OpenCV convention) cameras
    :param c2w: (N, 4)
    :return T (4,4) , scale (float)
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # (1) Rotate the world so that z+ is the up axis
    # we estimate the up axis by averaging the camera up axes
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        # In the unlikely case the original data has y+ up axis,
        # rotate 180-deg about x axis
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    #  R_align = np.eye(3) # DEBUG
    R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    # (2) Recenter the scene.
    if center_method == "focus":
        # find the closest point to the origin for each camera's center ray
        nearest = t + (fwds * -t).sum(-1)[:, None] * fwds
        translate = -np.median(nearest, axis=0)
    elif center_method == "poses":
        # use center of the camera positions
        translate = -np.median(t, axis=0)
    else:
        raise ValueError(f"Unknown center_method {center_method}")

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    # (3) Rescale the scene using camera distances
    scale_fn = np.max if strict_scaling else np.median
    inv_scale = scale_fn(np.linalg.norm(t + translate, axis=-1))
    if inv_scale == 0:
        inv_scale = 1.0
    scale = 1.0 / inv_scale
    transform[:3, :] *= scale

    return transform


def align_principle_axes(point_cloud):
    # Compute centroid
    centroid = np.median(point_cloud, axis=0)

    # Translate point cloud to centroid
    translated_point_cloud = point_cloud - centroid

    # Compute covariance matrix
    covariance_matrix = np.cov(translated_point_cloud, rowvar=False)

    # Compute eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvectors by eigenvalues (descending order) so that the z-axis
    # is the principal axis with the smallest eigenvalue.
    sort_indices = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, sort_indices]

    # Check orientation of eigenvectors. If the determinant of the eigenvectors is
    # negative, then we need to flip the sign of one of the eigenvectors.
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1

    # Create rotation matrix
    rotation_matrix = eigenvectors.T

    # Create SE(3) matrix (4x4 transformation matrix)
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = -rotation_matrix @ centroid

    return transform


def transform_points(matrix, points):
    """Transform points using a SE(4) matrix.

    Args:
        matrix: 4x4 SE(4) matrix
        points: Nx3 array of points

    Returns:
        Nx3 array of transformed points
    """
    assert matrix.shape == (4, 4)
    assert len(points.shape) == 2 and points.shape[1] == 3
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def transform_cameras(matrix, camtoworlds):
    """Transform cameras using a SE(4) matrix.

    Args:
        matrix: 4x4 SE(4) matrix
        camtoworlds: Nx4x4 array of camera-to-world matrices

    Returns:
        Nx4x4 array of transformed camera-to-world matrices
    """
    assert matrix.shape == (4, 4)
    assert len(camtoworlds.shape) == 3 and camtoworlds.shape[1:] == (4, 4)
    camtoworlds = np.einsum("nij, ki -> nkj", camtoworlds, matrix)
    scaling = np.linalg.norm(camtoworlds[:, 0, :3], axis=1)
    camtoworlds[:, :3, :3] = camtoworlds[:, :3, :3] / scaling[:, None, None]
    return camtoworlds


def normalize_scene(camtoworlds, points=None, camera_center_method="focus"):
    T1 = similarity_from_cameras(camtoworlds, center_method=camera_center_method)
    camtoworlds = transform_cameras(T1, camtoworlds)
    if points is not None:
        points = transform_points(T1, points)
        T2 = align_principle_axes(points)
        camtoworlds = transform_cameras(T2, camtoworlds)
        points = transform_points(T2, points)
        return camtoworlds, points, T2 @ T1
    else:
        return camtoworlds, T1
