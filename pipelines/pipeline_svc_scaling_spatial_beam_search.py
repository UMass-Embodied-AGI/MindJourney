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
from stable_virtual_camera.demo import svc_main, Model
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import pickle
import copy
os.environ["PYTORCH_SDP_FORCE_FALLBACK"] = "1"
from diffusers.utils import export_to_video
import quaternion
from pipeline_baseline import PipelineBase
import torch
from numpy import quaternion
import multiprocessing

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

def _run_one_candidate(action_list, magnitude,
                        img_path, step_dir, action_folder_name,
                        model_args,
                        model, # one model for each process
                        forward_size, turn_size):
    action_folder = os.path.join(step_dir, action_folder_name)
    os.makedirs(action_folder, exist_ok=True)

    # ---------- generate video ----------
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    video = np.tile(img_rgb[None, ...], (len(action_list) + 1, 1, 1, 1))
    export_to_video(video, os.path.join(action_folder, "video.mp4"), fps=1)

    # ---------- sample trajectory ----------
    pos   = np.zeros(3, dtype=float)
    theta = np.array([np.radians(-20.0), 0.0, 0.0], dtype=float)
    traj, traj_json = _build_trajectory(action_list, pos, theta,
                                        forward_size, turn_size)
    with open(os.path.join(action_folder, 'episode.pkl'), 'wb') as f:
        pickle.dump(traj, f)
    with open(os.path.join(action_folder, 'episode.json'), 'w') as f:
        json.dump(traj_json, f, indent=2)

    if action_list[0] == ActionSpace.MOVE_FORWARD:
        traj_prior = f"move-forward-{magnitude}"
    elif action_list[0] == ActionSpace.TURN_LEFT:
        traj_prior = f"turn-left-{magnitude}"
    else:
        traj_prior = f"turn-right-{magnitude}"

    svc_main(
        model=model,
        data_path=action_folder,
        task=model_args.task,
        replace_or_include_input=model_args.replace_or_include_input,
        traj_prior=traj_prior,
        cfg=model_args.cfg,
        guider=model_args.guider,
        L_short=model_args.L_short,
        num_targets=model_args.num_targets,
        use_traj_prior=model_args.use_traj_prior,
        output_path=action_folder,
        chunk_strategy=model_args.chunk_strategy,
        c2ws=traj['camera_extrinsic'],
    )
    return traj, traj_json


def _build_trajectory(action_list, pos, theta, forward_size, turn_size):
    try:
        trajectory = {
            'camera_pose': [],
            'camera_rotation': [],
            'camera_rotation_euler': [],
            'action': [],
            'camera_extrinsic': [],
        }
        trajectory_json = {
            'camera_pose': [],
            'camera_rotation_euler': [],
            'action': [],
        }

        def _append(a):
            r   = R.from_euler('xyz', theta, degrees=False)
            x, y, z, w = r.as_quat()
            quat = np.quaternion(w, x, y, z)
            trajectory['camera_pose'].append(np.round(pos, 4))
            trajectory['camera_rotation'].append(quat)
            trajectory['camera_rotation_euler'].append(np.round(theta, 4))
            trajectory['action'].append(a)
            trajectory_json['camera_pose'].append(np.round(pos, 4).tolist())
            trajectory_json['camera_rotation_euler'].append(np.round(theta, 4).tolist())
            trajectory_json['action'].append(a)
            rot_mat = r.as_matrix()
            c2w = np.eye(4)
            c2w[:3, :3] = rot_mat
            c2w[:3, 3] = pos
            trajectory['camera_extrinsic'].append(np.round(c2w, 6))

        _append(0)                                # first frame (no-op)
        for action in action_list:                # subsequent actions
            if action == ActionSpace.TURN_LEFT:
                theta[1] -= np.radians(turn_size)
            elif action == ActionSpace.TURN_RIGHT:
                theta[1] += np.radians(turn_size)
            elif action == ActionSpace.MOVE_FORWARD:
                dx = forward_size * np.sin(theta[1])
                dz = forward_size * np.cos(theta[1])
                pos += np.array([dx, 0.0, dz])
            else:
                raise ValueError(f"Unknown action: {action}")
            _append(action)

        return trajectory, trajectory_json

    except Exception as e:
        print(f"❌ Error in _run_one_candidate:")
        print(f"    action_list: {action_list}")
        return None, None


class SpatialVQAPipelineSVC(PipelineBase):
    """Class-based refactor of the original *source* script, keeping identical
    functionality but adopting the object-oriented structure used elsewhere
    in the code-base (see PipelineBase / PipelineSiyuan).
    """

    # ---------------------------------------------------------------------
    #  INIT
    # ---------------------------------------------------------------------
    def __init__(
        self,
   ):
        super().__init__()
        model_args = self.model_args
        self.global_model = Model()

    # ------------------------------------------------------------------
    #  PUBLIC ENTRY-POINT
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Main evaluation loop (mirrors the logic of the original script)."""
        for question in self.questions:
            # ------------------------------------------------------------------
            #  Quick filters & deduplication
            # ------------------------------------------------------------------
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

            # ------------------------------------------------------------------
            #  Set-up per-question output folder & initial image(s)
            # ------------------------------------------------------------------
            save_dir = os.path.join(self.model_args.output_dir, f"{qid}")

            os.makedirs(os.path.join(save_dir, f"step_0"), exist_ok=True)

            # --- primary image ----------------------------------------------------------
            primary_img_path = os.path.join(save_dir, "step_0", "img_0.png")
            img = cv2.imread(question["img_paths"][0])
            if self.model_args.vlm_model_name == "OpenGVLab/InternVL3-14B":
                # resize to 512x512
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            else:
                img = resize_to_short_side(img, target_short=512)
            cv2.imwrite(primary_img_path, img)

            # --- optional helper image --------------------------------------------------
            helper_img_path = None
            if len(question["img_paths"]) > 1:
                helper_img_path = os.path.join(save_dir, "step_0", "helper_img.png")
                helper = cv2.imread(question["img_paths"][1])
                if self.model_args.vlm_model_name == "OpenGVLab/InternVL3-14B":
                    helper = cv2.resize(helper, (512, 512), interpolation=cv2.INTER_LINEAR)
                else:
                    helper = resize_to_short_side(helper, target_short=512)
                cv2.imwrite(helper_img_path, helper)

            # ------------------------------------------------------------------
            #  Dialogue loop (LLM <-> environment)
            # ------------------------------------------------------------------
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

                # ----------------------------------------------------------
                #  Query World Model - simulate all actions
                # ----------------------------------------------------------
                top_scores = []
                helpful_store = []
                print("previous_action_sequences", previous_action_sequences)
                print("previous_action_lists", previous_action_lists)
                possible_actions = [f"move-forward {self.model_args.fixed_forward_magnitudes}", f"turn-left {self.model_args.fixed_rotation_magnitudes}", f"turn-right {self.model_args.fixed_rotation_magnitudes}"]
                action_candidates = self._simulate_all_actions(
                    image_path=primary_img_path,
                    step_idx=step,
                    actions=possible_actions,         # e.g. ["move-forward 0.75", "turn-left 30", "turn-right 30"]
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
                # print("number of action candidates:", len(temp_actions_store))
                top_scores_all = [None, None]
                for rank_i in range(2):
                    for _ in range(self.model_args.max_tries_gpt):
                        sys_prompt, content = self.vlm.format_prompt(prompt_type="prompt_scores",
                            question=question["question"],
                            answer_choices=question["answer_choices"],
                            images=[primary_img_path, helper_img_path] if len(question['img_paths']) > 1 else [primary_img_path],
                            action_consequences=temp_actions_store,
                            sys_prompt=system_prompts_for_double_search[rank_i],
                        )
                        response = self.vlm.run_prompt("prompt_scores", sys_prompt, content)
                        print("[LLM RESPONSE in choose n]", response)
                        scores = self._process_list(response)
                        # print("[RESULT after process]", scores)
                        if scores != "out of control":
                            break
                    self._dump_llm_interaction(save_dir, step, rank_i, question, response, result, action_list, magnitude)
                    # save top indexes and corresponding scores to top_candidates (scores are unsorted, ranked by index)
                    # in the format of [(index, score), ...]
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
                    # Add to helpful action consequences, which had the same format as action_candidates
                    if rank_i == 1:
                        for score, index in top_scores_all[rank_i]:
                            if index >= len(temp_actions_store):
                                print("index out of range")
                                continue
                            action_str, subaction_str, img_path = temp_actions_store[index]
                            # if score is smaller than 6, skip
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
                            num_steps = round(magnitude / self.model_args.sampling_interval_meter)
                            for _ in range(num_steps):
                                previous_action_list.append(ActionSpace.MOVE_FORWARD)
                            previous_action_sequence.append(raw_action.replace("move forward", "move-forward"))
                        elif "turn left" in raw_action:
                            num_steps = round(magnitude / self.model_args.sampling_interval_angle)
                            for _ in range(num_steps):
                                previous_action_list.append(ActionSpace.TURN_LEFT)
                            previous_action_sequence.append(raw_action.replace("turn left", "turn-left"))
                        elif "turn right" in raw_action:
                            num_steps = round(magnitude / self.model_args.sampling_interval_angle)
                            for _ in range(num_steps):
                                previous_action_list.append(ActionSpace.TURN_RIGHT)
                            previous_action_sequence.append(raw_action.replace("turn right", "turn-right"))
                    previous_action_sequences.append(previous_action_sequence)
                    previous_action_lists.append(previous_action_list)
                # sort helpful_store by score
                helpful_store.sort(key=lambda x: x[0], reverse=True)
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
                # please determine the last image by the order of the subaction's magnitude
                subaction_consequences = sorted(subaction_consequences.items(), key=lambda x: float(x[0].split(" ")[-2]))
                helpful_action_consequences[action_str] = dict(subaction_consequences)

            # ----------------------------------------------------------
            #  Query LLM - allow a few retries if response unparsable
            # ----------------------------------------------------------
            number_of_imaginations = 0
            for action_str in helpful_action_consequences.keys():
                number_of_imaginations += len(helpful_action_consequences[action_str])
            print("Number of imaginations:", number_of_imaginations)
            for _ in range(self.model_args.max_tries_gpt):
                sys_prompt, content = self.vlm.format_prompt(prompt_type="answer_scaling",
                    question=question["question"],
                    answer_choices=question["answer_choices"],
                    images=[primary_img_path, helper_img_path] if len(question['img_paths']) > 1 else [primary_img_path],
                    action_consequences=helpful_action_consequences,
                )
                response = self.vlm.run_prompt("answer_scaling", sys_prompt, content)
                print("[LLM RESPONSE]", response)
                result = self._process_answer(response, question)
                # print("[RESULT After Process]", result)
                if result != "out of control":
                    break
            
            # Persist raw exchange for debugging / auditing
            self._dump_llm_interaction(save_dir, None, None, question, response, result, action_list, magnitude)

            if result == "out of control":
                self.results["parsing_err_stats"]["answer"] += 1
                self.results["parsing_err_stats"]["answer_qid"].append(qid)
                result = "wrong"

            # ----------------------------------------------------------
            #  Terminal answer?  -> store + break
            # ----------------------------------------------------------
            if result in ("correct", "wrong"):
                self.results["progress"][question["question_type"]][result].append(qid)

            # ------------------------------------------------------------------
            #  Persist progress after each question
            # ------------------------------------------------------------------
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


    # ------------------------------------------------------------------
    #  LLM RESPONSE PARSING
    # ------------------------------------------------------------------
    def _process_answer(self, response: str, question: dict, fwd=0.075, turn=3):
        """Parse LLM response and map to (result, actions, magnitude)."""
        response_l = response.lower()
        try:
            if any(c.lower() in response_l for c in question["answer_choices"]):
                return "correct" if question["correct_answer"].lower() in response_l.split("\n")[-1] else "wrong"
        except Exception:
            pass
        return "out of control"
    
    def _process_score(self, response: str):
        """Parse LLM response to integer score."""
        try:
            return int(response)
        except Exception:
            pass
        return "out of control"
    
    def _process_list(self, response: str):
        """Parse LLM response to a list."""
        try:
            if "Output:" in response:
                response = response.split("Output:")[1]
            list_ = [int(i.strip()) for i in response.split(",")]
            return list_
        except Exception:
            pass
        return "out of control"
    
    def _process_bbox(self, response: str):
        """Parse LLM response to a bounding box: [(x1,y1), (x2,y2)]. response is in format (150,160):(180,220)."""
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

    # ------------------------------------------------------------------
    #  ACTION EXECUTION VIA SVC WORLD MODEL
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    #  LOGGING HELPERS
    # ------------------------------------------------------------------
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
    
    def _take_action_with_world_model(self, step_idx, img_path, action_lists, magnitudes, step_dir, action_folder_name_list, forward_size=0.25, turn_size=9, num_workers=1):
        """
        parallel inference
        `num_workers`
        """
        tasks = [(alist, mag, img_path, step_dir, action_folder_name,
                copy.deepcopy(self.model_args),
                self.global_model, 
                forward_size, turn_size)
                for alist, mag, action_folder_name in zip(action_lists, magnitudes, action_folder_name_list)]

        if num_workers is None:
            num_workers = min(len(tasks), os.cpu_count() or 1)

        with multiprocessing.get_context("spawn").Pool(num_workers) as pool:
            results = pool.starmap(_run_one_candidate, tasks)

        # all_trajectories, all_trajectories_json = zip(*results)
        # return all_trajectories, all_trajectories_json
        return
    
    def _simulate_all_actions(
        self,
        image_path: str,
        step_idx: int,
        actions: List[str],  # e.g. ["move-forward 0.75", "turn-left 30"]
        sampling_interval_angle: int,  # e.g. 10° → sample every 10, 20, 30° …
        sampling_interval_meter: float,  # e.g. 0.25 m → sample every 0.25 m …
        model_args,
        save_dir: str,
        previous_action_sequences: Optional[List[List[str]]] = None,
        previous_action_lists: Optional[List[List[int]]] = None,
        sequential: bool = False,  # <- currently unused but kept for backward-compatibility
    ) -> Dict[str, Dict[str, str]]:
        """Simulate a batch of candidate actions and return sampled frame paths.

        Parameters
        ----------
        image_path
            Path to the current RGB frame that represents the agent's egocentric view.
        step_idx
            Index of the current decision step (0-based).
        actions
            Human-readable action strings - MUST follow the pattern
            ``"<verb> <magnitude>"``, e.g. ``"turn-left 30"`` or ``"move-forward 0.75"``.
        sampling_interval_angle
            Angular step (in degrees) at which to sample frames **for turning actions**.
        sampling_interval_meter
            Linear step (in meters) at which to sample frames **for forward actions**.
        model_args
            Namespace / dataclass that carries meta-args for the world-model (must expose
            ``num_frames`` and ``frame_interval``).
        save_dir
            Root directory where all rendered videos & sampled frames will be stored.
        previous_action_sequences, previous_action_lists
            Optional history of actions already executed in this episode.  They are used
            to stitch *new* candidate actions onto *previous* ones so that a single video
            containing the full compound trajectory can be generated and sampled.
            - ``previous_action_sequences`` contains the *raw* strings, e.g.
            ``[["move-forward 0.5", "turn-right 30"], …]``.
            - ``previous_action_lists``     contains the *encoded* ActionSpace IDs that
            the low-level controller needs, same outer list structure as above.

        Returns
        -------
        Dict[str, Dict[str, str]]
            ``{ top_key → { sub_key → frame_path } }`` where
            - *top_key*  encodes the **action family** (e.g. "turn left").
            - *sub_key*  is an individual sample (e.g. "turn left 20 degrees").
            - *frame_path* is a PNG outside ``save_dir`` pointing to the sampled frame.
        """

        # ---------------------------------------------------------------------
        # 0. Book-keeping variables & hyper-parameters
        # ---------------------------------------------------------------------
        action_candidates: Dict[str, Dict[str, str]] = {}

        # Folder layout:  <save_dir>/step_<step_idx>/<action_folder>/pred.mp4
        step_dir = os.path.join(save_dir, f"step_{step_idx}")
        os.makedirs(step_dir, exist_ok=True)

        fixed_length: int = model_args.num_frames - 1  # frames per **new** action
        turn_size: int = model_args.frame_interval * 3  # degrees per frame when turning
        forward_size: float = model_args.frame_interval * (0.25 / 3)  # metres per frame

        # We accumulate the following four lists to run the world-model **once**
        # for all candidate branches - provides huge speed-ups on GPUs.
        action_lists: List[List[int]] = []          # low-level ActionSpace IDs
        magnitudes: List[float]       = []          # angle / distance for each branch
        top_key_list: List[str]       = []          # family key → first-level dict
        action_folder_name_list: List[str] = []     # filesystem folder for the branch
        prev_action_len_list: List[int] = []        # #frames of *prepended* history

        # ---------------------------------------------------------------------
        # 1. Parse *each* provided high-level action string
        # ---------------------------------------------------------------------
        print("[SpatialVQA] Simulating actions:", actions)
        for action_str in actions:
            # --- 1-A. Validate & split ------------------------------------------------
            tokens = action_str.strip().split()
            if len(tokens) != 2:
                print(f"[WARN] Skip invalid action '{action_str}' - expected '<verb> <mag>'.")
                continue

            raw_action, magnitude_s = tokens
            try:
                magnitude = float(magnitude_s)  # "30" → 30.0
            except ValueError:
                print(f"[WARN] Cannot parse magnitude in '{action_str}'.  Skipping…")
                continue

            # Helper to map the *verb* into ActionSpace & human-readable family key.
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
                # print("action_id:", action_id)
                # print("family_key:", family_key)
            except ValueError as exc:
                print(f"[WARN] {exc}.  Skipping…")
                continue

            # -----------------------------------------------------------------
            # 1-B. Expand *either* plain actions *or* prepend history branches
            # -----------------------------------------------------------------
            if previous_action_sequences is None:
                # --- No history ⇒ single branch --------------------------------
                action_list = [action_id] * fixed_length
                folder_name = f"{raw_action}_{magnitude:.2f}"
                prev_len = 0

                # «Register» this branch
                action_lists.append(action_list)
                magnitudes.append(magnitude)
                top_key_list.append(family_key)
                action_folder_name_list.append(folder_name)
                prev_action_len_list.append(prev_len)
                action_candidates.setdefault(family_key, {})
            else:
                # --- Fan-out: prepend each *history* and test the new action ----
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

        print(f"!!!prev_action_len_list:{prev_action_len_list}")

        # ---------------------------------------------------------------------
        # 2. Run *once* through the world-model to render *ALL* branches
        # ---------------------------------------------------------------------
        self._take_action_with_world_model(
            step_idx=step_idx,
            img_path=image_path,
            action_lists=action_lists,
            magnitudes=magnitudes,
            step_dir=step_dir,
            action_folder_name_list=action_folder_name_list,
            forward_size=forward_size,
            turn_size=turn_size,
            num_workers=self.model_args.max_inference_batch_size
        )

        # ---------------------------------------------------------------------
        # 3. Sample frames from the rendered videos at the desired intervals
        # ---------------------------------------------------------------------
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

            # Edge-case: corrupted or too-short videos
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < 2:
                print(f"[ERR ] Not enough frames in {video_path!r} - skip.")
                cap.release()
                continue

            # -----------------------------------------------------------------
            # Compute the frame indices we want to grab (domain-specific maths)
            # -----------------------------------------------------------------
            if action_list[-1] != ActionSpace.MOVE_FORWARD:  # ↻ turning
                arr = np.arange(0, magnitude + 1, turn_size)
                targets = np.arange(0, magnitude + 1, sampling_interval_angle)
            else:  # → moving forward
                arr = np.arange(0, magnitude + 1e-3, forward_size)
                targets = np.arange(0, magnitude + 1e-3, sampling_interval_meter)

            sampled_indices = [int(np.abs(arr - t).argmin()) for t in targets]
            print("!!!sampled_indices", sampled_indices)
            if self.world_model_type == "cogvideox" or self.world_model_type == None:
                sampled_indices = sampled_indices[1:]  # drop the first frame (identical to input)
            elif self.world_model_type == "svc":
                sampled_indices = [x * 2 for x in sampled_indices]
                sampled_indices = sampled_indices[:-1]
            print("!!!sampled_indices", sampled_indices)

            # -----------------------------------------------------------------
            # Read & save sampled frames ☑
            # -----------------------------------------------------------------
            for i, frame_idx in enumerate(sampled_indices, start=1):
                if self.world_model_type == "cogvideox" or self.world_model_type == None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + prev_len)
                elif self.world_model_type == "svc":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + prev_len*2)
                success, frame = cap.read()
                if not success:
                    print(f"[WARN] Cannot grab frame {frame_idx+prev_len} from {video_path!r}.")
                    continue

                # Derive human-readable *sub-key* and filename ------------------
                if action_list[-1] != ActionSpace.MOVE_FORWARD:  # turning
                    metric_val = i * sampling_interval_angle
                    fname = f"sample_{metric_val}.png"
                    sub_key = f"{family_key} {metric_val} degrees"
                else:  # forward
                    metric_val = i * sampling_interval_meter
                    fname = f"sample_{metric_val}.png"
                    sub_key = f"{family_key} {metric_val} meters"

                out_path = os.path.join(step_dir, folder_name, fname)
                cv2.imwrite(out_path, frame)
                action_candidates[family_key][sub_key] = out_path

            cap.release()

        return action_candidates


# -----------------------------------------------------------------------------
#  CLI ENTRY
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise pipeline & run
    pipeline = SpatialVQAPipelineSVC()
    print("!!!!!!!!!!!!!!!!")
    pipeline.run()
