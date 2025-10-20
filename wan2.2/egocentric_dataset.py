import os
import glob
import json
import pickle
import random
from typing import List

import numpy as np
import torch
from torchvision.transforms import v2
from PIL import Image
import imageio
import quaternion as nq  # for handling numpy.quaternion rotations


class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)


class EgocentricDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path: str,
        num_frames: int = 41,
        frame_interval: int = 1,
        height: int = 480,
        width: int = 832,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.base_path = base_path
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.height = height
        self.width = width
        self.rng = random.Random(seed)

        # Augmentation configuration (probabilities and magnitudes)
        self.p_reverse = 0.5
        self.p_brightness = 0.1
        self.max_brightness_delta = 0.1  # in normalized [-1, 1] space
        self.p_contrast = 0.1
        self.max_contrast_factor_delta = 0.2  # factor in [1-d, 1+d]
        self.p_saturation = 0.1
        self.max_saturation_factor_delta = 0.2  # factor in [1-d, 1+d]
        self.p_gaussian_noise = 0.1
        self.gaussian_noise_std = 0.03  # relative to normalized [-1, 1]

        # Episodes discovery with dataset-wide cache.
        cache_path = os.path.join(self.base_path, "episodes_cache.json")
        episodes: List[dict] = []
        loaded_from_cache = False
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    # Filter out entries whose directories no longer exist.
                    for item in data:
                        epi_path = item.get("path")
                        if isinstance(epi_path, str) and os.path.isdir(epi_path):
                            # Ensure the episode still has a video.mp4
                            episodes.append({
                                "path": epi_path,
                                "episode": item.get("episode", os.path.basename(epi_path)),
                            })
                loaded_from_cache = len(episodes) > 0
            except Exception:
                loaded_from_cache = False

        if not loaded_from_cache:
            # Discover episodes recursively. Keep only those with video.mp4
            for epi_dir in sorted(glob.glob(os.path.join(self.base_path, "**"), recursive=True)):
                if not os.path.isdir(epi_dir):
                    continue
                if os.path.abspath(epi_dir) == os.path.abspath(self.base_path):
                    continue
                video_path = os.path.join(epi_dir, "video.mp4")
                if not os.path.isfile(video_path):
                    continue
                episodes.append({
                    "path": os.path.abspath(epi_dir),
                    "episode": os.path.basename(epi_dir),
                })

            # Write cache for future runs (best-effort).
            try:
                with open(cache_path, "w") as f:
                    json.dump(episodes, f)
            except Exception:
                pass

        self.episodes = episodes

        # Frame processing to square then resize
        self.frame_process = v2.Compose([
            v2.Lambda(lambda img: self._center_square_crop(img)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _center_square_crop(self, img):
        from PIL import Image as _PILImage
        if isinstance(img, _PILImage.Image):
            w, h = img.size
            if w == h:
                return img
            m = min(w, h)
            left = (w - m) // 2
            top = (h - m) // 2
            right = left + m
            bottom = top + m
            return img.crop((left, top, right, bottom))
        if isinstance(img, torch.Tensor):
            if img.ndim == 3:
                if img.shape[0] in (1, 3):
                    c, h, w = img.shape
                    m = min(h, w)
                    hs = (h - m) // 2
                    ws = (w - m) // 2
                    return img[:, hs:hs + m, ws:ws + m]
                else:
                    h, w, c = img.shape
                    m = min(h, w)
                    hs = (h - m) // 2
                    ws = (w - m) // 2
                    return img[hs:hs + m, ws:ws + m, :]
        return img

    def __len__(self) -> int:
        return len(self.episodes)

    def _load_episode_meta(self, episode_path: str):
        new_pkl = os.path.join(episode_path, "episode_new.pkl")
        old_pkl = os.path.join(episode_path, "episode.pkl")
        pkl_path = new_pkl if os.path.exists(new_pkl) else old_pkl
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"No episode pickle found at {new_pkl} or {old_pkl}")
        with open(pkl_path, "rb") as f:
            episode = pickle.load(f)
        camera_poses = np.asarray(episode.get("camera_pose"))  # (T, 3)
        camera_rotations = np.asarray(episode.get("camera_rotation"))  # (T, ...)
        return camera_poses, camera_rotations

    def _poses_to_extrinsics(self, poses: np.ndarray, rotations: np.ndarray) -> np.ndarray:
        def quat_to_rotmat_wxyz(q_arr: np.ndarray) -> np.ndarray:
            w, x, y, z = q_arr
            ww, xx, yy, zz = w*w, x*x, y*y, z*z
            xy, xz, yz = x*y, x*z, y*z
            wx, wy, wz = w*x, w*y, w*z
            return np.array([
                [ww + xx - yy - zz, 2*(xy - wz),       2*(xz + wy)],
                [2*(xy + wz),       ww - xx + yy - zz, 2*(yz - wx)],
                [2*(xz - wy),       2*(yz + wx),       ww - xx - yy + zz],
            ], dtype=np.float32)

        T = poses.shape[0]
        c2w_all = []
        for t in range(T):
            c2w = np.eye(4, dtype=np.float32)
            rot = rotations[t]
            R = None
            if isinstance(rot, np.ndarray) and rot.shape == (3, 3):
                R = rot.astype(np.float32)
            else:
                if getattr(rot, "__class__", None).__name__ == "quaternion":
                    R = nq.as_rotation_matrix(rot).astype(np.float32)
                else:
                    rot_arr = np.asarray(rot)
                    if getattr(rot_arr, "dtype", None) is not None and str(rot_arr.dtype) == "quaternion":
                        R = nq.as_rotation_matrix(rot_arr.item()).astype(np.float32)
                    elif rot_arr.shape == (4,):
                        q = nq.quaternion(rot_arr[0], rot_arr[1], rot_arr[2], rot_arr[3])
                        R = nq.as_rotation_matrix(q).astype(np.float32)
                    else:
                        try:
                            R = quat_to_rotmat_wxyz(rot_arr.astype(np.float32))
                        except Exception:
                            raise TypeError(f"Unsupported rotation format at t={t}: type={type(rot)}, shape={getattr(rot, 'shape', None)}")

            c2w[:3, :3] = R
            c2w[:3, 3] = poses[t].astype(np.float32)
            c2w_all.append(c2w)
        return np.stack(c2w_all, axis=0)

    def _load_frames_from_video(self, video_path: str, frame_indices: List[int]):
        reader = imageio.get_reader(video_path)
        frames = []
        for idx in frame_indices:
            idx = max(0, min(idx, reader.count_frames() - 1))
            frame = reader.get_data(idx)
            frame = Image.fromarray(frame)
            frames.append(self.frame_process(frame))
        reader.close()
        return torch.stack(frames, dim=0)  # T, C, H, W

    def _load_frames_from_images(self, image_paths: List[str], frame_indices: List[int]):
        num_total = len(image_paths)
        frames = []
        for idx in frame_indices:
            idx = max(0, min(idx, num_total - 1))
            frame = Image.open(image_paths[idx]).convert("RGB")
            frames.append(self.frame_process(frame))
        return torch.stack(frames, dim=0)  # T, C, H, W

    def _sample_indices(self, total_frames: int) -> List[int]:
        if total_frames <= 0:
            return [0] * self.num_frames
        if total_frames <= self.num_frames:
            start_idx = 0
        else:
            start_idx = self.rng.randint(0, max(0, total_frames - 1 - (self.num_frames - 1) * self.frame_interval))
        idxs = [min(start_idx + i * self.frame_interval, total_frames - 1) for i in range(self.num_frames)]
        return idxs

    def _apply_photometric(self, frames: torch.Tensor) -> torch.Tensor:
        out = frames
        if self.rng.random() < self.p_brightness:
            delta = (2.0 * self.rng.random() - 1.0) * self.max_brightness_delta
            out = out + float(delta)
        if self.rng.random() < self.p_contrast:
            delta = (2.0 * self.rng.random() - 1.0) * self.max_contrast_factor_delta
            factor = 1.0 + float(delta)
            mean = out.mean(dim=(1, 2, 3), keepdim=True)
            out = (out - mean) * factor + mean
        if self.rng.random() < self.p_saturation:
            delta = (2.0 * self.rng.random() - 1.0) * self.max_saturation_factor_delta
            factor = 1.0 + float(delta)
            r, g, b = out[:, 0:1], out[:, 1:2], out[:, 2:3]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            gray = gray.repeat(1, 3, 1, 1)
            out = gray + factor * (out - gray)
        if self.rng.random() < self.p_gaussian_noise:
            noise = torch.randn_like(out) * self.gaussian_noise_std
            out = out + noise
        out = torch.clamp(out, -1.0, 1.0)
        return out

    def __getitem__(self, index: int):
        num_episodes = len(self.episodes)
        if num_episodes == 0:
            raise IndexError("EgocentricDataset contains no episodes")

        start_idx = index % num_episodes
        attempts = 0
        last_error = None

        while attempts < num_episodes:
            epi = self.episodes[(start_idx + attempts) % num_episodes]
            epi_path = epi["path"]
            try:
                poses, rotations = self._load_episode_meta(epi_path)
                if poses is None or rotations is None:
                    raise ValueError("episode.pkl missing required keys")
                if poses.shape[0] == 0:
                    raise ValueError("episode contains zero poses")
                T_total = poses.shape[0]
                frame_indices = self._sample_indices(T_total)

                frames = None
                video_candidates = [
                    os.path.join(epi_path, "video.mp4"),
                    os.path.join(epi_path, "rgb", "video.mp4"),
                ]
                video_path = next((p for p in video_candidates if os.path.exists(p)), None)
                if video_path is not None:
                    try:
                        frames = self._load_frames_from_video(video_path, frame_indices)
                    except Exception:
                        frames = None

                if frames is None:
                    rgb_list = sorted(glob.glob(os.path.join(epi_path, "rgb_*.png")))
                    if len(rgb_list) == 0:
                        rgb_list = sorted(glob.glob(os.path.join(epi_path, "rgb", "*.png")))
                    if len(rgb_list) == 0:
                        raise FileNotFoundError("No readable video.mp4 or RGB images")
                    frames = self._load_frames_from_images(rgb_list, frame_indices)

                c2w_all = self._poses_to_extrinsics(poses, rotations)
                c2w_sampled = c2w_all[frame_indices]

                orig_frames = frames
                orig_extrinsics = c2w_sampled

                static_frames = None
                base_video_candidates = [
                    os.path.join(epi_path, "base_video.mp4"),
                    os.path.join(epi_path, "rgb", "base_video.mp4"),
                ]
                base_video_path = next((p for p in base_video_candidates if os.path.exists(p)), None)
                if base_video_path is not None:
                    try:
                        static_frames = self._load_frames_from_video(base_video_path, frame_indices)
                    except Exception:
                        static_frames = None

                rng_state = self.rng.getstate()
                aug_frames = self._apply_photometric(orig_frames)
                if static_frames is not None:
                    self.rng.setstate(rng_state)
                    aug_static_frames = self._apply_photometric(static_frames)

                do_reverse = self.rng.random() < self.p_reverse
                if static_frames is not None:
                    if do_reverse:
                        actual_view = torch.flip(aug_frames, dims=[0])
                        static_view = torch.flip(aug_static_frames, dims=[0])
                        actual_extrinsics = orig_extrinsics[::-1]
                        cond_c2w = orig_extrinsics[-1]
                    else:
                        actual_view = aug_frames
                        static_view = aug_static_frames
                        actual_extrinsics = orig_extrinsics
                        cond_c2w = orig_extrinsics[0]
                    static_extrinsics = np.stack([cond_c2w for _ in range(self.num_frames)], axis=0)
                else:
                    if do_reverse:
                        actual_view = torch.flip(aug_frames, dims=[0])
                        actual_extrinsics = orig_extrinsics[::-1]
                        cond_frame = aug_frames[-1:]
                        cond_c2w = orig_extrinsics[-1]
                    else:
                        actual_view = aug_frames
                        actual_extrinsics = orig_extrinsics
                        cond_frame = aug_frames[0:1]
                        cond_c2w = orig_extrinsics[0]
                    static_view = cond_frame.repeat(self.num_frames, 1, 1, 1)
                    static_extrinsics = np.stack([cond_c2w for _ in range(self.num_frames)], axis=0)

                combined_video = torch.cat([actual_view, static_view], dim=0)
                combined_video = combined_video.permute(1, 0, 2, 3).contiguous()

                inv_cond = np.linalg.inv(cond_c2w)
                rel_actual = np.stack([inv_cond @ m for m in actual_extrinsics], axis=0)
                rel_static = np.stack([inv_cond @ m for m in static_extrinsics], axis=0)

                def mat4_to_3x4(m):
                    return m[:3, :]
                rel_actual_3x4 = np.stack([mat4_to_3x4(m) for m in rel_actual], axis=0)
                rel_static_3x4 = np.stack([mat4_to_3x4(m) for m in rel_static], axis=0)
                camera_poses = np.concatenate([rel_actual_3x4, rel_static_3x4], axis=0)
                camera_poses = torch.from_numpy(camera_poses.reshape(self.num_frames * 2, 12)).to(torch.float32)

                text = "A random egocentric room tour"

                return {
                    "text": text,
                    "video": combined_video,  # C, 2*T, H, W
                    "camera": camera_poses,   # (2*T, 12)
                    "path": epi_path,
                    "episode": epi.get("episode", os.path.basename(epi_path)),
                }
            except Exception as e:
                last_error = e
                attempts += 1
                continue

        raise RuntimeError(f"All episodes failed to load. Last error: {last_error}")


