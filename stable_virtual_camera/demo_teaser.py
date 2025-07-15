import glob
import os
import os.path as osp

import fire
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from .seva.data_io import get_parser
from .seva.eval import (
    IS_TORCH_NIGHTLY,
    compute_relative_inds,
    create_transforms_simple,
    infer_prior_inds,
    infer_prior_stats,
    run_one_scene,
)
from .seva.geometry import (
    generate_interpolated_path,
    generate_spiral_path,
    get_arc_horizontal_w2cs,
    get_default_intrinsics,
    get_lookat,
    get_preset_pose_fov,
    DEFAULT_FOV_RAD,
)
from .seva.model import SGMWrapper
from .seva.modules.autoencoder import AutoEncoder
from .seva.modules.conditioner import CLIPConditioner
from .seva.sampling import DDPMDiscretization, DiscreteDenoiser
from .seva.utils import load_model

device = "cuda:0"


# Constants.
WORK_DIR = "work_dirs/demo"

# if IS_TORCH_NIGHTLY:
#     COMPILE = True
#     os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
#     os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
# else:
#     COMPILE = False

# MODEL = SGMWrapper(load_model(device="cpu", verbose=True).eval()).to(device)
# AE = AutoEncoder(chunk_size=1).to(device)
# CONDITIONER = CLIPConditioner().to(device)
# DISCRETIZATION = DDPMDiscretization()
# DENOISER = DiscreteDenoiser(discretization=DISCRETIZATION, num_idx=1000, device=device)
VERSION_DICT = {
    "H": 576,
    "W": 576,
    "T": 17,
    "C": 4,
    "f": 8,
    "options": {},
}

# if COMPILE:
#     MODEL = torch.compile(MODEL, dynamic=False)
#     CONDITIONER = torch.compile(CONDITIONER, dynamic=False)
#     AE = torch.compile(AE, dynamic=False)

class Model:
    def __init__(self):
        
        if IS_TORCH_NIGHTLY:
            COMPILE = True
            os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
            os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
        else:
            COMPILE = False

        self.model = SGMWrapper(load_model(device="cpu", verbose=True).eval()).to(device)
        self.ae = AutoEncoder(chunk_size=1).to(device)
        self.conditioner = CLIPConditioner().to(device)
        DISCRETIZATION = DDPMDiscretization()
        self.denoiser = DiscreteDenoiser(discretization=DISCRETIZATION, num_idx=1000, device=device)

        if COMPILE:
            self.model = torch.compile(MODEL, dynamic=False)
            self.conditioner = torch.compile(CONDITIONER, dynamic=False)
            self.ae = torch.compile(AE, dynamic=False)


def parse_task(
    task,
    scene,
    num_inputs,
    T,
    version_dict,
):
    options = version_dict["options"]

    anchor_indices = None
    anchor_c2ws = None
    anchor_Ks = None

    c2ws = options.get("c2ws")
    c2ws = np.stack(c2ws, axis=0)
    print("!!!!",c2ws.shape)
    print(c2ws[1])
    

    if task == "img2trajvid_s-prob":
        if num_inputs is not None:
            assert (
                num_inputs == 1
            ), "Task `img2trajvid_s-prob` only support 1-view conditioning..."
        else:
            num_inputs = 1
        num_targets = options.get("num_targets", T - 1)
        num_anchors = infer_prior_stats(
            T,
            num_inputs,
            num_total_frames=num_targets,
            version_dict=version_dict,
        )

        input_indices = [0]
        anchor_indices = np.linspace(1, num_targets, num_anchors).tolist()

        all_imgs_path = [scene] + [None] * num_targets
        fovs = np.full((num_targets+1,), DEFAULT_FOV_RAD)

        # import math
        # angle_deg = -20
        # angle_rad = math.radians(angle_deg)

        # # Rotation matrix for pitch down around X-axis
        # R = torch.tensor([
        #     [1, 0, 0],
        #     [0, math.cos(angle_rad), -math.sin(angle_rad)],
        #     [0, math.sin(angle_rad),  math.cos(angle_rad)]
        # ])

        # # Combine with translation (no translation here)
        # w2c = torch.eye(4)
        # w2c[:3, :3] = R

        # c2ws, fovs = get_preset_pose_fov(
        #     option=options.get("traj_prior", "orbit"),
        #     num_frames=num_targets + 1,
        #     start_w2c=w2c,
        #     look_at=torch.Tensor([0, 0, 10]),
        # )
        # print("????",c2ws.shape)
        # print(c2ws[1])

        with Image.open(scene) as img:
            W, H = img.size
            aspect_ratio = W / H
        Ks = get_default_intrinsics(fovs, aspect_ratio=aspect_ratio)  # unormalized
        Ks[:, :2] *= (
            torch.tensor([W, H]).reshape(1, -1, 1).repeat(Ks.shape[0], 1, 1)
        )  # normalized
        Ks = Ks.numpy()

        anchor_c2ws = c2ws[[round(ind) for ind in anchor_indices]]
        anchor_Ks = Ks[[round(ind) for ind in anchor_indices]]

    else:
        parser = get_parser(
            parser_type="reconfusion",
            data_dir=scene,
            normalize=False,
        )
        all_imgs_path = parser.image_paths
        c2ws = parser.camtoworlds
        camera_ids = parser.camera_ids
        Ks = np.concatenate([parser.Ks_dict[cam_id][None] for cam_id in camera_ids], 0)

        if num_inputs is None:
            assert len(parser.splits_per_num_input_frames.keys()) == 1
            num_inputs = list(parser.splits_per_num_input_frames.keys())[0]
            split_dict = parser.splits_per_num_input_frames[num_inputs]  # type: ignore
        elif isinstance(num_inputs, str):
            split_dict = parser.splits_per_num_input_frames[num_inputs]  # type: ignore
            num_inputs = int(num_inputs.split("-")[0])  # for example 1_from32
        else:
            split_dict = parser.splits_per_num_input_frames[num_inputs]  # type: ignore

        num_targets = len(split_dict["test_ids"])

        if task == "img2img":
            # Note in this setting, we should refrain from using all the other camera
            # info except ones from sampled_indices, and most importantly, the order.
            num_anchors = infer_prior_stats(
                T,
                num_inputs,
                num_total_frames=num_targets,
                version_dict=version_dict,
            )

            sampled_indices = np.sort(
                np.array(split_dict["train_ids"] + split_dict["test_ids"])
            )  # we always sort all indices first

            traj_prior = options.get("traj_prior", None)
            if traj_prior == "spiral":
                assert parser.bounds is not None
                anchor_c2ws = generate_spiral_path(
                    c2ws[sampled_indices] @ np.diagflat([1, -1, -1, 1]),
                    parser.bounds[sampled_indices],
                    n_frames=num_anchors + 1,
                    n_rots=2,
                    zrate=0.5,
                    endpoint=False,
                )[1:] @ np.diagflat([1, -1, -1, 1])
            elif traj_prior == "interpolated":
                assert num_inputs > 1
                anchor_c2ws = generate_interpolated_path(
                    c2ws[split_dict["train_ids"], :3],
                    round((num_anchors + 1) / (num_inputs - 1)),
                    endpoint=False,
                )[1 : num_anchors + 1]
            elif traj_prior == "orbit":
                c2ws_th = torch.as_tensor(c2ws)
                lookat = get_lookat(
                    c2ws_th[sampled_indices, :3, 3],
                    c2ws_th[sampled_indices, :3, 2],
                )
                anchor_c2ws = torch.linalg.inv(
                    get_arc_horizontal_w2cs(
                        torch.linalg.inv(c2ws_th[split_dict["train_ids"][0]]),
                        lookat,
                        -F.normalize(
                            c2ws_th[split_dict["train_ids"]][:, :3, 1].mean(0),
                            dim=-1,
                        ),
                        num_frames=num_anchors + 1,
                        endpoint=False,
                    )
                ).numpy()[1:, :3]
            else:
                anchor_c2ws = None
            # anchor_Ks is default to be the first from target_Ks

            all_imgs_path = [all_imgs_path[i] for i in sampled_indices]
            c2ws = c2ws[sampled_indices]
            Ks = Ks[sampled_indices]

            # absolute to relative indices
            input_indices = compute_relative_inds(
                sampled_indices,
                np.array(split_dict["train_ids"]),
            )
            anchor_indices = np.arange(
                sampled_indices.shape[0],
                sampled_indices.shape[0] + num_anchors,
            ).tolist()  # the order has no meaning here

        elif task == "img2vid":
            num_targets = len(all_imgs_path) - num_inputs
            num_anchors = infer_prior_stats(
                T,
                num_inputs,
                num_total_frames=num_targets,
                version_dict=version_dict,
            )

            input_indices = split_dict["train_ids"]
            anchor_indices = infer_prior_inds(
                c2ws,
                num_prior_frames=num_anchors,
                input_frame_indices=input_indices,
                options=options,
            ).tolist()
            num_anchors = len(anchor_indices)
            anchor_c2ws = c2ws[anchor_indices, :3]
            anchor_Ks = Ks[anchor_indices]

        elif task == "img2trajvid":
            num_anchors = infer_prior_stats(
                T,
                num_inputs,
                num_total_frames=num_targets,
                version_dict=version_dict,
            )

            target_c2ws = c2ws[split_dict["test_ids"], :3]
            target_Ks = Ks[split_dict["test_ids"]]
            anchor_c2ws = target_c2ws[
                np.linspace(0, num_targets - 1, num_anchors).round().astype(np.int64)
            ]
            anchor_Ks = target_Ks[
                np.linspace(0, num_targets - 1, num_anchors).round().astype(np.int64)
            ]

            sampled_indices = split_dict["train_ids"] + split_dict["test_ids"]
            all_imgs_path = [all_imgs_path[i] for i in sampled_indices]
            c2ws = c2ws[sampled_indices]
            Ks = Ks[sampled_indices]

            input_indices = np.arange(num_inputs).tolist()
            anchor_indices = np.linspace(
                num_inputs, num_inputs + num_targets - 1, num_anchors
            ).tolist()

        else:
            raise ValueError(f"Unknown task: {task}")

    return (
        all_imgs_path,
        num_inputs,
        num_targets,
        input_indices,
        anchor_indices,
        torch.tensor(c2ws[:, :3]).float(),
        torch.tensor(Ks).float(),
        (torch.tensor(anchor_c2ws[:, :3]).float() if anchor_c2ws is not None else None),
        (torch.tensor(anchor_Ks).float() if anchor_Ks is not None else None),
    )


def svc_main(
    model,
    data_path,
    data_items=None,
    task="img2img",
    save_subdir="",
    H=None,
    W=None,
    T=None,
    use_traj_prior=False,
    output_path=None,
    seed=23,
    img_path=None,
    **overwrite_options,
):
    if H is not None:
        VERSION_DICT["H"] = H
    if W is not None:
        VERSION_DICT["W"] = W
    if T is not None:
        VERSION_DICT["T"] = [int(t) for t in T.split(",")] if isinstance(T, str) else T

    MODEL, AE, CONDITIONER, DENOISER = model.model, model.ae, model.conditioner, model.denoiser

    options = VERSION_DICT["options"]
    options["chunk_strategy"] = "interp"
    options["video_save_fps"] = 30.0
    options["beta_linear_start"] = 5e-6
    options["log_snr_shift"] = 2.4
    options["guider_types"] = 1
    options["cfg"] = 2.0
    options["camera_scale"] = 2.0
    options["num_steps"] = 20
    options["cfg_min"] = 1.2
    options["encoding_t"] = 1
    options["decoding_t"] = 1
    options["num_inputs"] = None
    options["seed"] = 23
    options["replace_or_include_input"] = True
    options.update(overwrite_options)

    num_inputs = options["num_inputs"]


    scenes = [img_path]
    # scenes = sorted(scenes)
    # scenes = scenes[-1:]
    print("scenes", scenes)

    for scene in scenes:
        print("!!!",scene)
        if output_path != None:
            save_path_scene = output_path
        else:    
            save_path_scene = os.path.join(
            WORK_DIR, task, save_subdir, os.path.splitext(os.path.basename(scene))[0]
        )
        if options.get("skip_saved", False) and os.path.exists(
            os.path.join(save_path_scene, "transforms.json")
        ):
            print(f"Skipping {scene} as it is already sampled.")
            continue

        # parse_task -> infer_prior_stats modifies VERSION_DICT["T"] in-place.
        (
            all_imgs_path,
            num_inputs,
            num_targets,
            input_indices,
            anchor_indices,
            c2ws,
            Ks,
            anchor_c2ws,
            anchor_Ks,
        ) = parse_task(
            task,
            scene,
            num_inputs,
            VERSION_DICT["T"],
            VERSION_DICT,
        )
        assert num_inputs is not None
        # Create image conditioning.
        image_cond = {
            "img": all_imgs_path,
            "input_indices": input_indices,
            "prior_indices": anchor_indices,
        }
        # Create camera conditioning.
        camera_cond = {
            "c2w": c2ws.clone(),
            "K": Ks.clone(),
            "input_indices": list(range(num_inputs + num_targets)),
        }
        # run_one_scene -> transform_img_and_K modifies VERSION_DICT["H"] and VERSION_DICT["W"] in-place.
        video_path_generator = run_one_scene(
            task,
            VERSION_DICT,  # H, W maybe updated in run_one_scene
            model=MODEL,
            ae=AE,
            conditioner=CONDITIONER,
            denoiser=DENOISER,
            image_cond=image_cond,
            camera_cond=camera_cond,
            save_path=save_path_scene,
            use_traj_prior=use_traj_prior,
            traj_prior_Ks=anchor_Ks,
            traj_prior_c2ws=anchor_c2ws,
            seed=seed,  # to ensure sampled video can be reproduced in regardless of start and i
        )
        for _ in video_path_generator:
            pass

        # Convert from OpenCV to OpenGL camera format.
        # c2ws = c2ws @ torch.tensor(np.diag([1, -1, -1, 1])).float()
        # img_paths = sorted(glob.glob(osp.join(save_path_scene, "*.png")))
        # if len(img_paths) != len(c2ws):
        #     input_img_paths = sorted(
        #         glob.glob(osp.join(save_path_scene, "input", "*.png"))
        #     )
        #     assert len(img_paths) == num_targets
        #     assert len(input_img_paths) == num_inputs
        #     assert c2ws.shape[0] == num_inputs + num_targets
        #     target_indices = [i for i in range(c2ws.shape[0]) if i not in input_indices]
        #     img_paths = [
        #         input_img_paths[input_indices.index(i)]
        #         if i in input_indices
        #         else img_paths[target_indices.index(i)]
        #         for i in range(c2ws.shape[0])
        #     ]
        # create_transforms_simple(
        #     save_path=save_path_scene,
        #     img_paths=img_paths,
        #     img_whs=np.array([VERSION_DICT["W"], VERSION_DICT["H"]])[None].repeat(
        #         num_inputs + num_targets, 0
        #     ),
        #     c2ws=c2ws,
        #     Ks=Ks,
        # )


if __name__ == "__main__":
    fire.Fire(svc_main)
