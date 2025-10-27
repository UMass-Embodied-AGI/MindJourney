from utils.api import ChatAPI, AzureConfig
from utils.prompt_formatting import *
from tqdm import tqdm
import argparse
import json
import random
import os
import cv2
import sys
from typing import Dict, List, Optional
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import pickle
import copy
from diffusers.utils import export_to_video
from pipeline_baseline import PipelineBase
import torch
import time
import types
from torchvision.transforms import v2
from PIL import Image
from diffsynth import save_video


def resize_to_short_side(img, target_short=512):
    h, w = img.shape[:2]
    if min(h, w) == target_short:          # already the right size
        return img
    scale = target_short / float(min(h, w))
    new_w, new_h = int(math.ceil(w * scale)), int(math.ceil(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


class ActionSpace:
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3


class SpatialVQAPipeline(PipelineBase):
    """WAN2.2-based world model variant, keeping outer logic unchanged."""

    def __init__(
        self,
    ):
        super().__init__()
        # TODO
        self.model_args.seed = 42
        model_args = self.model_args

        if model_args.seed is not None:
            set_seed(model_args.seed)

        # ------------------------------------------------------------------
        # Load WAN2.2 pipeline from local wan2.2/ directory (package shimming)
        # ------------------------------------------------------------------
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        wan_pkg_fs_path = os.path.join(project_root, "wan2.2")
        shim_name = "wan2_2"
        if shim_name not in sys.modules:
            pkg = types.ModuleType(shim_name)
            pkg.__path__ = [wan_pkg_fs_path]
            sys.modules[shim_name] = pkg

        from wan2_2.wan_video_mvva import WanVideoReCamMasterPipeline
        from wan2_2.models import ModelConfig

        # Minimal default configs (no extra CLI changes)
        model_id = getattr(model_args, "model_id", "Wan-AI/Wan2.2-TI2V-5B")
        model_configs = [
            ModelConfig(
                local_model_path="checkpoints",
                model_id=model_id,
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
                offload_device="cpu",
            ),
            ModelConfig(
                local_model_path="checkpoints",
                model_id=model_id,
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16*.pth",
                offload_device="cpu",
            ),
            ModelConfig(
                local_model_path="checkpoints",
                model_id=model_id,
                origin_file_pattern="Wan*_VAE.pth",
                offload_device="cpu",
            ),
        ]
        tokenizer_config = ModelConfig(
            local_model_path="checkpoints", model_id=model_id, origin_file_pattern="google/*"
        )
        self.new_pipeline = WanVideoReCamMasterPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            redirect_common_files=False,
        )

        # Initialize camera encoders on DIT blocks
        dim = self.new_pipeline.dit.blocks[0].self_attn.q.weight.shape[0]
        for block in self.new_pipeline.dit.blocks:
            if not hasattr(block, "cam_encoder"):
                block.cam_encoder = torch.nn.Linear(12, dim)
                torch.nn.init.kaiming_uniform_(block.cam_encoder.weight, a=math.sqrt(5))
                if block.cam_encoder.bias is not None:
                    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(block.cam_encoder.weight)
                    bound = 1 / math.sqrt(fan_in)
                    torch.nn.init.uniform_(block.cam_encoder.bias, -bound, bound)
            if not hasattr(block, "projector"):
                block.projector = torch.nn.Linear(dim, dim)
                torch.nn.init.eye_(block.projector.weight)
                if block.projector.bias is not None:
                    torch.nn.init.zeros_(block.projector.bias)

        # Optional: load finetuned ckpt via env var if provided (robust to nesting/prefixes)
        ckpt_path = os.environ.get("WAN_CKPT_PATH", None)
        if ckpt_path is not None and os.path.isfile(ckpt_path):
            raw = torch.load(ckpt_path, map_location="cpu")
            sd = raw.get("state_dict", raw)
            # If keys are prefixed with 'module.' or 'dit.', strip them for dit load
            def strip_prefix(d, prefix):
                return {k[len(prefix):]: v for k, v in d.items() if k.startswith(prefix)}
            if any(k.startswith("module.") for k in sd.keys()):
                sd = strip_prefix(sd, "module.")
            if any(k.startswith("dit.") for k in sd.keys()):
                sd = strip_prefix(sd, "dit.")
            try:
                missing, unexpected = self.new_pipeline.dit.load_state_dict(sd, strict=False)
                if missing or unexpected:
                    print(f"[INFO] Loaded ckpt with missing={len(missing)}, unexpected={len(unexpected)} keys from {ckpt_path}.")
            except Exception as e:
                print(f"[WARN] Failed to load finetuned ckpt from {ckpt_path}: {e}. Proceeding with base weights.")

        self.new_pipeline.to("cuda")
        self.new_pipeline.to(dtype=torch.bfloat16)
        self.timing = {}

    def run(self) -> None:
        for question in self.questions:
            if question["question_type"] in ["other"]:
                print(f"[SpatialVQA] Skipping question {question['database_idx']} - not a spatial VQA task.")
                self.results["skip_indices"].append(question["database_idx"])
                self.save_results()
                continue

            if self.model_args.question_type != "None" and question["question_type"] != self.model_args.question_type:
                print(f"[SpatialVQA] Skipping question {question['database_idx']} - not a {self.model_args.question_type} task.")
                self.results["skip_indices"].append(question["database_idx"])
                self.save_results()
                continue

            if len(question["img_paths"]) > self.model_args.max_images:
                print(f"[SpatialVQA] Skipping question {question['database_idx']} - only one image supported.")
                self.results["skip_indices"].append(question["database_idx"])
                self.save_results()
                continue

            qid = question["database_idx"]
            if (
                qid in self.results["skip_indices"]
                or any(qid in result["correct"] for result in self.results["progress"].values())
                or any(qid in result["wrong"] for result in self.results["progress"].values())
            ):
                print(f"[SpatialVQA] Skipping already processed question {qid}.")
                continue

            os.makedirs(os.path.join(self.model_args.output_dir, f"{qid}"), exist_ok=True)

            save_dir = os.path.join(self.model_args.output_dir, f"{qid}")
            os.makedirs(os.path.join(save_dir, f"step_0"), exist_ok=True)

            primary_img_path = os.path.join(save_dir, "step_0", "img_0.png")
            img = cv2.imread(question["img_paths"][0])
            if self.model_args.vlm_model_name == "OpenGVLab/InternVL3-14B":
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            # else:
            #     img = resize_to_short_side(img, target_short=512)
            cv2.imwrite(primary_img_path, img)

            helper_img_path = None
            if len(question["img_paths"]) > 1:
                helper_img_path = os.path.join(save_dir, "step_0", "helper_img.png")
                helper = cv2.imread(question["img_paths"][1])
                if self.model_args.vlm_model_name == "OpenGVLab/InternVL3-14B":
                    helper = cv2.resize(helper, (512, 512), interpolation=cv2.INTER_LINEAR)
                # else:
                #     helper = resize_to_short_side(helper, target_short=512)
                cv2.imwrite(helper_img_path, helper)

            response, result, action_list, magnitude = None, "out of control", [], None
            previous_action_sequences = None
            previous_action_lists = None
            helpful_action_consequences = {}
            system_prompts_for_double_search = [
                (
                    "You are an AI assistant designed to help with spatial reasoning in a 3D indoor scene. "
                    "You must analyze any provided images and score imagined images based on how suitable they are for exploring these action consequences in order to answer the question from the choices.\n\n"
                    "Rules:\n"
                    "1. You'll be provided with images (including imagined images), a question, and a set of answer choices. You should score all imagined images.\n"
                    "2. You should output a list of scores from 0 to 9, separated by ','. For example: Output: 3,5,2,9,0,1\n"
                ),
                (
                    "You are an AI assistant designed to help with spatial reasoning in a 3D indoor scene. "
                    "You must analyze any provided images and score imagined images based on how helpful they are for answering the questions. Hint: They may not be correct for answering the questions, but they will be helpful for excluding the wrong answers. The scores should also consider the image quality. If the image quality is very bad, it should receive a low score. Otherwise, the score should be augmented.\n\n"
                    "Rules:\n"
                    "1. You'll be provided with images (including imagined images), a question, and a set of answer choices. You should score all imagined images.\n"
                    "2. You should output a list of scores from 0 to 9, separated by ','. For example: Output: 3,5,2,9,0,1\n"
                )
            ]
            for step in range(self.model_args.max_steps_per_question):
                print(f"\n[SpatialVQA] ===== Step {step} for QID {qid} =====")

                top_scores = []
                helpful_store = []
                print("previous_action_sequences", previous_action_sequences)
                print("previous_action_lists", previous_action_lists)
                possible_actions = [f"move-forward {self.model_args.fixed_forward_magnitudes}", f"turn-left {self.model_args.fixed_rotation_magnitudes}", f"turn-right {self.model_args.fixed_rotation_magnitudes}"]

                action_candidates = self._simulate_all_actions(
                    image_path=primary_img_path,
                    step_idx=step,
                    actions=possible_actions,
                    sampling_interval_angle=self.model_args.sampling_interval_angle,
                    sampling_interval_meter=self.model_args.sampling_interval_meter,
                    model_args=self.model_args,
                    save_dir=save_dir,
                    previous_action_sequences=previous_action_sequences if previous_action_sequences is not None else None,
                    previous_action_lists=previous_action_lists if previous_action_lists is not None else None,
                )
                temp_actions_store = []
                for action_str, subaction_consequences in action_candidates.items():
                    for subaction_str, img_path in subaction_consequences.items():
                        temp_actions_store.append((action_str, subaction_str, img_path))
                if len(temp_actions_store) == 0:
                    print("No action candidates found. Goes to Q&A.")
                    break
                top_scores_all = [None, None]
                self.timing[f'vlm_timing1_step{step}'] = []
                for rank_i in range(2):
                    for _ in range(self.model_args.max_tries_gpt):
                        start_time = time.time()
                        sys_prompt, content = self.vlm.format_prompt(prompt_type="prompt_scores",
                            question=question["question"],
                            answer_choices=question["answer_choices"],
                            images=[primary_img_path, helper_img_path] if len(question['img_paths']) > 1 else [primary_img_path],
                            action_consequences=temp_actions_store,
                            sys_prompt=system_prompts_for_double_search[rank_i],
                        )
                        response = self.vlm.run_prompt("prompt_scores", sys_prompt, content)
                        end_time = time.time()
                        self.timing[f'vlm_timing1_step{step}'].append(end_time - start_time)
                        print("[LLM RESPONSE in choose n]", response)
                        scores = self._process_list(response)
                        if scores != "out of control":
                            break
                    self._dump_llm_interaction(save_dir, step, rank_i, question, response, result, action_list, magnitude)
                    if scores != "out of control":
                        top_scores_all[rank_i] = [(score, index) for index, score in enumerate(scores)]
                        top_scores_all[rank_i].sort(key=lambda x: x[0], reverse=True)
                    else:
                        self.results["parsing_err_stats"]["scores"] += 1
                        if rank_i == 1:
                            self.results["parsing_err_stats"]["scores_qid"].append(qid)
                        scores = [random.randint(0, 5) for _ in range(len(temp_actions_store))]
                        print("sampled scores:", scores)
                        top_scores_all[rank_i] = [(score, index) for index, score in enumerate(scores)]
                        top_scores_all[rank_i].sort(key=lambda x: x[0], reverse=True)
                    if rank_i == 1:
                        for score, index in top_scores_all[rank_i]:
                            if index >= len(temp_actions_store):
                                print("index out of range")
                                continue
                            action_str, subaction_str, img_path = temp_actions_store[index]
                            if score < self.model_args.helpful_score_threshold:
                                continue
                            if img_path not in [helpful[3] for helpful in helpful_store]:
                                helpful_store.append((score, action_str, subaction_str, img_path))

                previous_action_sequences = []
                previous_action_lists = []
                for score, index in top_scores_all[0][:self.model_args.num_beams]:
                    if index >= len(temp_actions_store):
                        print("index out of range")
                        continue
                    if score < self.model_args.exploration_score_threshold:
                        continue
                    action_str, subaction_str, img_path = temp_actions_store[index]
                    previous_action_list = []
                    previous_action_sequence = []
                    raw_extracted_previous_action_sequence = subaction_str.split(", and then ")[0].split(", ")
                    if ", and then " in subaction_str:
                        raw_extracted_previous_action_sequence.append(subaction_str.split(", and then ")[1])
                    for raw_action in raw_extracted_previous_action_sequence:
                        magnitude = float(raw_action.split(" ")[-2])
                        if "move forward" in raw_action:
                            num_steps = round(magnitude / (self.model_args.frame_interval * (0.25 / 3)))
                            for _ in range(num_steps):
                                previous_action_list.append(ActionSpace.MOVE_FORWARD)
                            previous_action_sequence.append(raw_action.replace("move forward", "move-forward"))
                        elif "turn left" in raw_action:
                            num_steps = round(magnitude / (self.model_args.frame_interval * 3))
                            for _ in range(num_steps):
                                previous_action_list.append(ActionSpace.TURN_LEFT)
                            previous_action_sequence.append(raw_action.replace("turn left", "turn-left"))
                        elif "turn right" in raw_action:
                            num_steps = round(magnitude / (self.model_args.frame_interval * 3))
                            for _ in range(num_steps):
                                previous_action_list.append(ActionSpace.TURN_RIGHT)
                            previous_action_sequence.append(raw_action.replace("turn right", "turn-right"))
                    previous_action_sequences.append(previous_action_sequence)
                    previous_action_lists.append(previous_action_list)
                helpful_store.sort(key=lambda x: x[0], reverse=True)
                print("helpful_store", helpful_store)
                number_of_successful_additions = 0
                for score, action_str, subaction_str, img_path in helpful_store:
                    if number_of_successful_additions < self.model_args.num_top_candidates:
                        if action_str not in helpful_action_consequences:
                            helpful_action_consequences[action_str] = {}
                        if subaction_str not in helpful_action_consequences[action_str]:
                            helpful_action_consequences[action_str][subaction_str] = img_path
                            number_of_successful_additions += 1
                    else:
                        break

            for action_str, subaction_consequences in helpful_action_consequences.items():
                subaction_consequences = sorted(subaction_consequences.items(), key=lambda x: float(x[0].split(" ")[-2]))
                helpful_action_consequences[action_str] = dict(subaction_consequences)

            number_of_imaginations = 0
            for action_str in helpful_action_consequences.keys():
                number_of_imaginations += len(helpful_action_consequences[action_str])
            print("Number of imaginations:", number_of_imaginations)
            self.timing[f'vlm_timing2_step{step}'] = []
            for _ in range(self.model_args.max_tries_gpt):
                start_time = time.time()
                sys_prompt, content = self.vlm.format_prompt(prompt_type="answer_scaling",
                    question=question["question"],
                    answer_choices=question["answer_choices"],
                    images=[primary_img_path, helper_img_path] if len(question['img_paths']) > 1 else [primary_img_path],
                    action_consequences=helpful_action_consequences,
                )
                response = self.vlm.run_prompt("answer_scaling", sys_prompt, content)
                end_time = time.time()
                self.timing[f'vlm_timing2_step{step}'].append(end_time - start_time)
                print("[LLM RESPONSE]", response)
                result = self._process_answer(response, question)
                if result != "out of control":
                    break

            self._dump_llm_interaction(save_dir, None, None, question, response, result, action_list, magnitude)

            if result == "out of control":
                self.results["parsing_err_stats"]["answer"] += 1
                self.results["parsing_err_stats"]["answer_qid"].append(qid)
                result = "wrong"

            if result in ("correct", "wrong"):
                self.results["progress"][question["question_type"]][result].append(qid)

            all_types = self.results["progress"].keys()
            correct_total = sum(len(self.results["progress"][t]["correct"]) for t in all_types)
            wrong_total = sum(len(self.results["progress"][t]["wrong"]) for t in all_types)
            self.results["accuracy"]["all"] = correct_total / (correct_total + wrong_total)
            for t in all_types:
                if len(self.results["progress"][t]["correct"]) + len(self.results["progress"][t]["wrong"]) != 0:
                    self.results["accuracy"]["types"][t] = len(self.results["progress"][t]["correct"]) / (
                        len(self.results["progress"][t]["correct"]) + len(self.results["progress"][t]["wrong"])
                    )
                else:
                    self.results["accuracy"]["types"][t] = None
            self.results["current"] = f"{correct_total + wrong_total + len(self.results['skip_indices'])} / {len(self.questions)}"
            self.save_results()

            json.dump(self.timing, open(os.path.join(save_dir, "timing.json"), "w"), indent=4)

    def _process_answer(self, response: str, question: dict, fwd=0.075, turn=3):
        response_l = response.lower()
        try:
            if any(c.lower() in response_l for c in question["answer_choices"]):
                return "correct" if question["correct_answer"].lower() in response_l.split("\n")[-1] else "wrong"
        except Exception:
            pass
        return "out of control"

    def _process_score(self, response: str):
        try:
            return int(response)
        except Exception:
            pass
        return "out of control"

    def _process_list(self, response: str):
        try:
            if "Output:" in response:
                response = response.split("Output:")[1]
            list_ = [int(i.strip()) for i in response.split(",")]
            return list_
        except Exception:
            pass
        return "out of control"

    def _process_bbox(self, response: str):
        try:
            if "Output:" in response:
                response = response.split("Output:")[1]
            if "None" in response:
                return None
            list_= []
            coordinates = response.split(":")
            for coordinate in coordinates:
                coordinate = coordinate.strip()[1:-1].split(',')
                list_.append((int(coordinate[0]), int(coordinate[1])))
            return list_
        except Exception:
            pass
        return "out of control"

    def bbox_mask(self, bbox, image):
        (x1, y1), (x2, y2) = bbox
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1
        return mask

    def get_action_command_and_magnitude(self, action_str):
        return action_str.split(" ")[0], float(action_str.split(" ")[1])

    def _dump_llm_interaction(self, save_dir, step, double_search_step, question, response, result, actions, magnitude, auxiliary=None, content=None):
        prompt = copy.deepcopy(self.vlm.curr_prompt)
        log = {
            "auxiliary": auxiliary if auxiliary is not None else None,
            "question": question,
            "result": result,
            "action_list": actions,
            "magnitude": magnitude,
            "llm_response": response,
            "prompt": prompt,
        }
        if step is not None and double_search_step is not None:
            step_dir = os.path.join(save_dir, f"step_{step}")
            os.makedirs(step_dir, exist_ok=True)
            with open(os.path.join(step_dir, f"gpt_{double_search_step}.json"), "w") as f:
                json.dump(log, f, indent=2)
        else:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "gpt.json"), "w") as f:
                json.dump(log, f, indent=2)

    def _take_action_with_world_model(self, step_idx, img_path, action_lists, magnitudes, step_dir, action_folder_name_list, forward_size=0.25, turn_size=9):
        model_args = self.model_args
        pipe      = self.new_pipeline

        if action_lists is None or len(action_lists) == 0:
            return

        # Load the starting image and preprocess identically to EgocentricDataset
        height = getattr(model_args, "height", 384)
        width  = getattr(model_args, "width", 384)
        num_frames = getattr(model_args, "num_frames", 9)
        cfg_scale = getattr(model_args, "guidance_scale", 5.0)
        num_inference_steps = getattr(model_args, "inference_step", 50)
        max_bs = getattr(model_args, "max_inference_batch_size", 1)

        def _center_square_crop_pil(img: Image.Image) -> Image.Image:
            w, h = img.size
            if w == h:
                return img
            m = min(w, h)
            left = (w - m) // 2
            top = (h - m) // 2
            right = left + m
            bottom = top + m
            return img.crop((left, top, right, bottom))

        frame_process = v2.Compose([
            v2.Lambda(lambda im: _center_square_crop_pil(im)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        pil_img = Image.open(img_path).convert("RGB")
        img_t = frame_process(pil_img)  # C,H,W in [-1, 1]

        # Prebuild all branch inputs
        branch_folders = []
        branch_cameras = []
        branch_sources = []

        for action_folder_name, action_list, magnitude in zip(action_folder_name_list, action_lists, magnitudes):
            action_folder = os.path.join(step_dir, action_folder_name)
            os.makedirs(action_folder, exist_ok=True)
            branch_folders.append(action_folder)

            # Build trajectory poses (Euler yaw changes; forward in camera's local Z)
            pos   = np.zeros(3, dtype=float)
            theta = np.array([np.radians(-10.0), 0.0, 0.0], dtype=float)

            c2w_list = []
            for i, action in enumerate([0] + action_list):
                if action == ActionSpace.TURN_LEFT:
                    theta[1] += np.radians(turn_size)
                elif action == ActionSpace.TURN_RIGHT:
                    theta[1] -= np.radians(turn_size)
                elif action == ActionSpace.MOVE_FORWARD:
                    dx = -forward_size * np.sin(theta[1])
                    dz = -forward_size * np.cos(theta[1])
                    pos += np.array([dx, 0.0, dz])
                Rmat = R.from_euler('xyz', theta, degrees=False).as_matrix()
                c2w = np.eye(4, dtype=np.float32)
                c2w[:3, :3] = Rmat.astype(np.float32)
                c2w[:3, 3]  = pos.astype(np.float32)
                c2w_list.append(c2w)

            cond_c2w = c2w_list[0]
            inv_cond = np.linalg.inv(cond_c2w)
            rel_actual = [inv_cond @ m for m in c2w_list[:num_frames]]
            rel_static = [np.eye(4, dtype=np.float32) for _ in range(num_frames)]

            def mat4_to_3x4(m):
                return m[:3, :]
            rel_actual_3x4 = np.stack([mat4_to_3x4(m) for m in rel_actual], axis=0)
            rel_static_3x4 = np.stack([mat4_to_3x4(m) for m in rel_static], axis=0)
            camera_poses = np.concatenate([rel_actual_3x4, rel_static_3x4], axis=0)  # (2T, 3, 4)
            camera_poses = torch.from_numpy(camera_poses.reshape(num_frames * 2, 12)).to(torch.float32)
            camera_poses = torch.cat([camera_poses[:num_frames][::4], camera_poses[num_frames:][::4]], dim=0)  # (2*(T/4), 12)
            branch_cameras.append(camera_poses)

            # Source video: repeat the first image T times (already normalized to [-1, 1])
            source_video = img_t.unsqueeze(1).repeat(1, num_frames, 1, 1)  # C,T,H,W
            branch_sources.append(source_video)

        # Run in batches
        for start in range(0, len(branch_folders), max_bs):
            end = min(start + max_bs, len(branch_folders))
            batched_folders = branch_folders[start:end]
            cam_batch = torch.stack(branch_cameras[start:end], dim=0)  # B, 2*(T/4), 12
            src_batch = torch.stack(branch_sources[start:end], dim=0)  # B, C, T, H, W
            src_batch = src_batch  # already [-1,1]

            # Add batch dimension to match pipeline expectations (B,C,T,H,W)
            # and ensure types are FP32 prior to WAN pipeline which casts internally.
            src_batch = src_batch.to(torch.float32)

            videos = pipe(
                prompt="A random egocentric room tour",
                negative_prompt="blurry, blurry, blurry, unclear, low quality, messy",
                source_video=src_batch,
                target_camera=cam_batch.unsqueeze(0) if cam_batch.dim() == 2 else cam_batch,  # (B, N, 12)
                height=height,
                width=width,
                num_frames=num_frames,
                cfg_scale=cfg_scale,
                num_inference_steps=num_inference_steps,
                seed=model_args.seed or 0,
                tiled=True,
                tile_size=(height // 16, width // 16),
                tile_stride=(height // 32, width // 32),
            )

            # Save outputs
            if isinstance(videos, list) and len(videos) == (end - start) and isinstance(videos[0], list):
                for frames, out_folder in zip(videos, batched_folders):
                    video_np = np.stack([np.array(fr) for fr in frames], axis=0)
                    save_video(video_np, os.path.join(out_folder, "pred.mp4"), fps=1, quality=5)
            else:
                # Fallback: single video returned (shouldn't happen when B>1)
                frames = videos
                video_np = np.stack([np.array(fr) for fr in frames], axis=0)
                save_video(video_np, os.path.join(batched_folders[0], "pred.mp4"), fps=1, quality=5)

        return

    def _simulate_all_actions(
        self,
        image_path: str,
        step_idx: int,
        actions: List[str],
        sampling_interval_angle: int,
        sampling_interval_meter: float,
        model_args,
        save_dir: str,
        previous_action_sequences: Optional[List[List[str]]] = None,
        previous_action_lists: Optional[List[List[int]]] = None,
        sequential: bool = False,
    ) -> Dict[str, Dict[str, str]]:
        action_candidates: Dict[str, Dict[str, str]] = {}
        step_dir = os.path.join(save_dir, f"step_{step_idx}")
        os.makedirs(step_dir, exist_ok=True)

        fixed_length: int = model_args.num_frames - 1
        turn_size: int = model_args.frame_interval * 3
        forward_size: float = model_args.frame_interval * (0.25 / 3)

        action_lists: List[List[int]] = []
        magnitudes: List[float]       = []
        top_key_list: List[str]       = []
        action_folder_name_list: List[str] = []
        prev_action_len_list: List[int] = []

        print("[SpatialVQA] Simulating actions:", actions)
        for action_str in actions:
            tokens = action_str.strip().split()
            if len(tokens) != 2:
                print(f"[WARN] Skip invalid action '{action_str}' - expected '<verb> <mag>'.")
                continue

            raw_action, magnitude_s = tokens
            try:
                magnitude = float(magnitude_s)
            except ValueError:
                print(f"[WARN] Cannot parse magnitude in '{action_str}'.  Skipping…")
                continue

            def _verb_to_ids(verb: str):
                if verb == "turn-left":
                    return ActionSpace.TURN_LEFT, "turn left"
                if verb == "turn-right":
                    return ActionSpace.TURN_RIGHT, "turn right"
                if verb == "move-forward":
                    return ActionSpace.MOVE_FORWARD, "move forward"
                raise ValueError(f"Unsupported action verb '{verb}'.")

            try:
                action_id, family_key = _verb_to_ids(raw_action)
            except ValueError as exc:
                print(f"[WARN] {exc}.  Skipping…")
                continue

            if previous_action_sequences is None:
                action_list = [action_id] * fixed_length
                folder_name = f"{raw_action}_{magnitude:.2f}"
                prev_len = 0

                action_lists.append(action_list)
                magnitudes.append(magnitude)
                top_key_list.append(family_key)
                action_folder_name_list.append(folder_name)
                prev_action_len_list.append(prev_len)
                action_candidates.setdefault(family_key, {})
            else:
                for hist_raw, hist_ids in zip(previous_action_sequences, previous_action_lists):
                    curr_action_command, curr_action_magnitude = self.get_action_command_and_magnitude(action_str)
                    last_action_command, last_action_magnitude = self.get_action_command_and_magnitude(hist_raw[-1])
                    if curr_action_command == "turn-left" and last_action_command == "turn-right":
                        print(f"[WARN] Skip invalid action '{action_str}' - cannot turn left after turning right.")
                        continue
                    if curr_action_command == "turn-right" and last_action_command == "turn-left":
                        print(f"[WARN] Skip invalid action '{action_str}' - cannot turn right after turning left.")
                        continue
                    if curr_action_command == last_action_command:
                        if curr_action_command in ("turn-left", "turn-right") and (curr_action_magnitude + last_action_magnitude) > self.model_args.max_turn_angle:
                            print(f"[WARN] Skip invalid action '{action_str}' - turning too far.")
                            continue
                        if curr_action_command == "move-forward" and (curr_action_magnitude + last_action_magnitude) > self.model_args.max_forward_distance:
                            print(f"[WARN] Skip invalid action '{action_str}' - moving too far.")
                            continue

                    if curr_action_command == last_action_command:
                        new_magnitude = curr_action_magnitude + last_action_magnitude
                        new_hist_raw = hist_raw[:-1]
                        if len(new_hist_raw) > 0:
                            hist_folder_prefix = "_".join(act.replace(" ", "_") for act in new_hist_raw) + "_"
                            hist_key_prefix   = ", ".join(act.replace("-", " ") for act in new_hist_raw) + ", and then "
                        else:
                            hist_folder_prefix = ""
                            hist_key_prefix   = ""
                        new_action_list = hist_ids + [action_id] * (fixed_length - len(hist_ids))
                        folder_name     = f"{hist_folder_prefix}{raw_action}_{new_magnitude:.2f}_meters" if curr_action_command == "move-forward" else f"{hist_folder_prefix}{raw_action}_{new_magnitude:.2f}_degrees"
                        family_full_key = hist_key_prefix + family_key
                        action_lists.append(new_action_list)
                        magnitudes.append(new_magnitude)
                        top_key_list.append(family_full_key)
                        action_folder_name_list.append(folder_name)
                        prev_action_len_list.append(len(hist_ids))
                        action_candidates.setdefault(family_full_key, {})
                    else:
                        hist_folder_prefix = "_".join(act.replace(" ", "_") for act in hist_raw) + "_"
                        hist_key_prefix   = ", ".join(act.replace("-", " ") for act in hist_raw) + ", and then "
                        new_action_list = hist_ids + [action_id] * (fixed_length - len(hist_ids))
                        folder_name     = f"{hist_folder_prefix}{raw_action}_{magnitude:.2f}_meters" if curr_action_command == "move-forward" else f"{hist_folder_prefix}{raw_action}_{magnitude:.2f}_degrees"
                        family_full_key = hist_key_prefix + family_key
                        action_lists.append(new_action_list)
                        magnitudes.append(magnitude)
                        top_key_list.append(family_full_key)
                        action_folder_name_list.append(folder_name)
                        prev_action_len_list.append(len(hist_ids))
                        action_candidates.setdefault(family_full_key, {})
        print(f"[INFO] Simulating {len(action_lists)} action branches…")
        # Run WAN world-model per branch
        self._take_action_with_world_model(
            step_idx=step_idx,
            img_path=image_path,
            action_lists=action_lists,
            magnitudes=magnitudes,
            step_dir=step_dir,
            action_folder_name_list=action_folder_name_list,
            forward_size=forward_size,
            turn_size=turn_size,
        )

        # Sample frames from generated videos
        for family_key, folder_name, prev_len, action_list, magnitude in zip(
            top_key_list,
            action_folder_name_list,
            prev_action_len_list,
            action_lists,
            magnitudes,
        ):
            video_path = os.path.join(step_dir, folder_name, "pred.mp4")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[ERR ] Could not open {video_path!r} - skip this branch.")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < 2:
                print(f"[ERR ] Not enough frames in {video_path!r} - skip.")
                cap.release()
                continue

            # Determine the contiguous length of the immediately preceding segment
            # that has the same action as the current one (e.g., continuing a turn-left).
            current_action_id = action_list[-1]
            prev_segment_len = 0
            if prev_len > 0 and action_list[prev_len - 1] == current_action_id:
                j = prev_len - 1
                while j >= 0 and action_list[j] == current_action_id:
                    prev_segment_len += 1
                    j -= 1
            segment_start_len = prev_len - prev_segment_len

            if current_action_id != ActionSpace.MOVE_FORWARD:
                # Turning segment
                step_size = turn_size
                prev_magnitude = prev_segment_len * step_size
                arr = np.arange(0, magnitude + 1, step_size)
                targets = np.arange(prev_magnitude + sampling_interval_angle, magnitude + 1, sampling_interval_angle)
            else:
                # Forward segment
                step_size = forward_size
                prev_distance = prev_segment_len * step_size
                arr = np.arange(0, magnitude + 1e-3, step_size)
                targets = np.arange(prev_distance + sampling_interval_meter, magnitude + 1e-3, sampling_interval_meter)

            # Map desired targets to closest frame indices in the generated video
            sampled_indices = [int(np.abs(arr - t).argmin()) for t in targets]

            for t, frame_idx in zip(targets, sampled_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + segment_start_len)
                success, frame = cap.read()
                if not success:
                    print(f"[WARN] Cannot grab frame {frame_idx+segment_start_len} from {video_path!r}.")
                    continue

                if current_action_id != ActionSpace.MOVE_FORWARD:
                    metric_val = int(round(t))
                    fname = f"sample_{metric_val}.png"
                    sub_key = f"{family_key} {metric_val} degrees"
                else:
                    metric_val = float(np.round(t, 3))
                    fname = f"sample_{metric_val}.png"
                    sub_key = f"{family_key} {metric_val} meters"

                out_path = os.path.join(step_dir, folder_name, fname)
                cv2.imwrite(out_path, frame)
                action_candidates[family_key][sub_key] = out_path

            cap.release()

        return action_candidates


if __name__ == "__main__":
    pipeline = SpatialVQAPipeline()
    pipeline.run()


