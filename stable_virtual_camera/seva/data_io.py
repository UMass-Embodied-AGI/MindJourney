import json
import os
import os.path as osp
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import cv2
import imageio.v3 as iio
import numpy as np
import torch

from stable_virtual_camera.seva.geometry import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, _, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


class BaseParser(object):
    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: Optional[int] = 8,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        self.image_names: List[str] = []  # (num_images,)
        self.image_paths: List[str] = []  # (num_images,)
        self.camtoworlds: np.ndarray = np.zeros((0, 4, 4))  # (num_images, 4, 4)
        self.camera_ids: List[int] = []  # (num_images,)
        self.Ks_dict: Dict[int, np.ndarray] = {}  # Dict of camera_id -> K
        self.params_dict: Dict[int, np.ndarray] = {}  # Dict of camera_id -> params
        self.imsize_dict: Dict[
            int, Tuple[int, int]
        ] = {}  # Dict of camera_id -> (width, height)
        self.points: np.ndarray = np.zeros((0, 3))  # (num_points, 3)
        self.points_err: np.ndarray = np.zeros((0,))  # (num_points,)
        self.points_rgb: np.ndarray = np.zeros((0, 3))  # (num_points, 3)
        self.point_indices: Dict[str, np.ndarray] = {}  # Dict of image_name -> (M,)
        self.transform: np.ndarray = np.zeros((4, 4))  # (4, 4)

        self.mapx_dict: Dict[int, np.ndarray] = {}  # Dict of camera_id -> (H, W)
        self.mapy_dict: Dict[int, np.ndarray] = {}  # Dict of camera_id -> (H, W)
        self.roi_undist_dict: Dict[int, Tuple[int, int, int, int]] = (
            dict()
        )  # Dict of camera_id -> (x, y, w, h)
        self.scene_scale: float = 1.0


class DirectParser(BaseParser):
    def __init__(
        self,
        imgs: List[np.ndarray],
        c2ws: np.ndarray,
        Ks: np.ndarray,
        points: Optional[np.ndarray] = None,
        points_rgb: Optional[np.ndarray] = None,  # uint8
        mono_disps: Optional[List[np.ndarray]] = None,
        normalize: bool = False,
        test_every: Optional[int] = None,
    ):
        super().__init__("", 1, normalize, test_every)

        self.image_names = [f"{i:06d}" for i in range(len(imgs))]
        self.image_paths = ["null" for _ in range(len(imgs))]
        self.camtoworlds = c2ws
        self.camera_ids = [i for i in range(len(imgs))]
        self.Ks_dict = {i: K for i, K in enumerate(Ks)}
        self.imsize_dict = {
            i: (img.shape[1], img.shape[0]) for i, img in enumerate(imgs)
        }
        if points is not None:
            self.points = points
            assert points_rgb is not None
            self.points_rgb = points_rgb
            self.points_err = np.zeros((len(points),))

        self.imgs = imgs
        self.mono_disps = mono_disps

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(self.camtoworlds)
            self.camtoworlds = transform_cameras(T1, self.camtoworlds)

            if points is not None:
                self.points = transform_points(T1, self.points)
                T2 = align_principle_axes(self.points)
                self.camtoworlds = transform_cameras(T2, self.camtoworlds)
                self.points = transform_points(T2, self.points)
            else:
                T2 = np.eye(4)

            self.transform = T2 @ T1
        else:
            self.transform = np.eye(4)

        # size of the scene measured by cameras
        camera_locations = self.camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)


class COLMAPParser(BaseParser):
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: Optional[int] = 8,
        image_folder: str = "images",
        colmap_folder: str = "sparse/0",
    ):
        super().__init__(data_dir, factor, normalize, test_every)

        colmap_dir = os.path.join(data_dir, colmap_folder)
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        try:
            from pycolmap import SceneManager
        except ImportError:
            raise ImportError(
                "Please install pycolmap to use the data parsers: "
                "  `pip install git+https://github.com/jensenz-sai/pycolmap.git@543266bc316df2fe407b3a33d454b310b1641042`"
            )

        manager = SceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        # Extract extrinsic matrices in world-to-camera format.
        imdata = manager.images
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.camera_type
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            assert (
                camtype == "perspective"  # type: ignore
            ), f"Only support perspective camera model, got {type_}"

            params_dict[camera_id] = params  # type: ignore

            # image size
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)

        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")
        if not (type_ == 0 or type_ == 1):  # type: ignore
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [imdata[k].name for k in imdata]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load images.
        if factor > 1:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, image_folder)
        image_dir = os.path.join(data_dir, image_folder + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        # 3D points and {image_name -> [point_idx]}
        points = manager.points3D.astype(np.float32)  # type: ignore
        points_err = manager.point3D_errors.astype(np.float32)  # type: ignore
        points_rgb = manager.point3D_colors.astype(np.uint8)  # type: ignore
        point_indices = dict()

        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principle_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]
            K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                K, params, (width, height), 0
            )
            mapx, mapy = cv2.initUndistortRectifyMap(
                K,
                params,
                None,
                K_undist,
                (width, height),
                cv2.CV_32FC1,  # type: ignore
            )
            self.Ks_dict[camera_id] = K_undist
            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.roi_undist_dict[camera_id] = roi_undist  # type: ignore

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)


class ReconfusionParser(BaseParser):
    def __init__(self, data_dir: str, normalize: bool = False):
        super().__init__(data_dir, 1, normalize, test_every=None)

        def get_num(p):
            return p.split("_")[-1].removesuffix(".json")

        splits_per_num_input_frames = {}
        num_input_frames = [
            int(get_num(p)) if get_num(p).isdigit() else get_num(p)
            for p in sorted(glob(osp.join(data_dir, "train_test_split_*.json")))
        ]
        for num_input_frames in num_input_frames:
            with open(
                osp.join(
                    data_dir,
                    f"train_test_split_{num_input_frames}.json",
                )
            ) as f:
                splits_per_num_input_frames[num_input_frames] = json.load(f)
        self.splits_per_num_input_frames = splits_per_num_input_frames

        with open(osp.join(data_dir, "transforms.json")) as f:
            metadata = json.load(f)

        image_names, image_paths, camtoworlds = [], [], []
        for frame in metadata["frames"]:
            if frame["file_path"] is None:
                image_path = image_name = None
            else:
                image_path = osp.join(data_dir, frame["file_path"])
                image_name = osp.basename(image_path)
            image_paths.append(image_path)
            image_names.append(image_name)
            camtoworld = np.array(frame["transform_matrix"])
            if "applied_transform" in metadata:
                applied_transform = np.concatenate(
                    [metadata["applied_transform"], [[0, 0, 0, 1]]], axis=0
                )
                camtoworld = applied_transform @ camtoworld
            camtoworlds.append(camtoworld)
        camtoworlds = np.array(camtoworlds)
        camtoworlds[:, :, [1, 2]] *= -1

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            self.transform = T1
        else:
            self.transform = np.eye(4)

        self.image_names = image_names
        self.image_paths = image_paths
        self.camtoworlds = camtoworlds
        self.camera_ids = list(range(len(image_paths)))
        self.Ks_dict = {
            i: np.array(
                [
                    [
                        metadata.get("fl_x", frame.get("fl_x", None)),
                        0.0,
                        metadata.get("cx", frame.get("cx", None)),
                    ],
                    [
                        0.0,
                        metadata.get("fl_y", frame.get("fl_y", None)),
                        metadata.get("cy", frame.get("cy", None)),
                    ],
                    [0.0, 0.0, 1.0],
                ]
            )
            for i, frame in enumerate(metadata["frames"])
        }
        self.imsize_dict = {
            i: (
                metadata.get("w", frame.get("w", None)),
                metadata.get("h", frame.get("h", None)),
            )
            for i, frame in enumerate(metadata["frames"])
        }
        # When num_input_frames is None, use all frames for both training and
        # testing.
        # self.splits_per_num_input_frames[None] = {
        #     "train_ids": list(range(len(image_paths))),
        #     "test_ids": list(range(len(image_paths))),
        # }

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)

        self.bounds = None
        if osp.exists(osp.join(data_dir, "bounds.npy")):
            self.bounds = np.load(osp.join(data_dir, "bounds.npy"))
            scaling = np.linalg.norm(self.transform[0, :3])
            self.bounds = self.bounds / scaling


class Dataset(torch.utils.data.Dataset):
    """A simple dataset class."""

    def __init__(
        self,
        parser: BaseParser,
        split: str = "train",
        num_input_frames: Optional[int] = None,
        patch_size: Optional[int] = None,
        load_depths: bool = False,
        load_mono_disps: bool = False,
    ):
        self.parser = parser
        self.split = split
        self.num_input_frames = num_input_frames
        self.patch_size = patch_size
        self.load_depths = load_depths
        self.load_mono_disps = load_mono_disps
        if load_mono_disps:
            assert isinstance(parser, DirectParser)
            assert parser.mono_disps is not None
        if isinstance(parser, ReconfusionParser):
            ids_per_split = parser.splits_per_num_input_frames[num_input_frames]
            self.indices = ids_per_split[
                "train_ids" if split == "train" else "test_ids"
            ]
        else:
            indices = np.arange(len(self.parser.image_names))
            if split == "train":
                self.indices = (
                    indices[indices % self.parser.test_every != 0]
                    if self.parser.test_every is not None
                    else indices
                )
            else:
                self.indices = (
                    indices[indices % self.parser.test_every == 0]
                    if self.parser.test_every is not None
                    else indices
                )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        if isinstance(self.parser, DirectParser):
            image = self.parser.imgs[index]
        else:
            image = iio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict.get(camera_id, None)
        camtoworlds = self.parser.camtoworlds[index]

        x, y, w, h = 0, 0, image.shape[1], image.shape[0]
        if params is not None and len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
        }

        if self.load_depths:
            # projected points to image plane to get depths
            worldtocams = np.linalg.inv(camtoworlds)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices[image_name]
            points_world = self.parser.points[point_indices]
            points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T
            points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
            depths = points_cam[:, 2]  # (M,)
            if self.patch_size is not None:
                points[:, 0] -= x
                points[:, 1] -= y
            # filter out points outside the image
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            points = points[selector]
            depths = depths[selector]
            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()
        if self.load_mono_disps:
            data["mono_disps"] = torch.from_numpy(self.parser.mono_disps[index]).float()  # type: ignore

        return data


def get_parser(parser_type: str, **kwargs) -> BaseParser:
    if parser_type == "colmap":
        parser = COLMAPParser(**kwargs)
    elif parser_type == "direct":
        parser = DirectParser(**kwargs)
    elif parser_type == "reconfusion":
        parser = ReconfusionParser(**kwargs)
    else:
        raise ValueError(f"Unknown parser type: {parser_type}")
    return parser
