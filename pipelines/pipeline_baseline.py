from utils.vlm_wrapper import VLMWrapper
from utils.prompt_formatting import SYS, BASELINE_PROMPT
from tqdm import tqdm
import argparse
import json
import random
import os
import cv2
import sys
from utils.args import get_svc_args
from decord import VideoReader, cpu
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import pickle
from diffusers.utils import export_to_video
import copy

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

class PipelineBase:

    def __init__(
        self,
    ):  
        self.world_model_type = os.getenv('WORLD_MODEL_TYPE')
        self.question_database_type = os.getenv('QUESTION_DATABASE_TYPE')
        self.model_args = get_svc_args()
        print("model_args:", self.model_args)
        self.prompt = BASELINE_PROMPT

        num_questions = self.model_args.num_questions
        
        if self.model_args.question_type == "None":
            input_file = os.path.join(self.model_args.input_dir, f"{self.model_args.split}.json")
        else:
            input_file = os.path.join(self.model_args.input_dir, f"{self.model_args.split}_{self.model_args.question_type}.json")
        if self.model_args.scaling_strategy is not None:
            self.model_args.output_dir += f"_{self.model_args.scaling_strategy}"
            os.makedirs(self.model_args.output_dir, exist_ok=True)
        if self.model_args.num_question_chunks > 1:
            self.model_args.output_dir += f"_qc{self.model_args.num_question_chunks}"
            os.makedirs(self.model_args.output_dir, exist_ok=True)
            self.model_args.output_dir = os.path.join(self.model_args.output_dir, f"question_chunk_{self.model_args.question_chunk_idx}")
            os.makedirs(self.model_args.output_dir, exist_ok=True)

        self.questions = self.select_questions(input_file, seed=10, num_questions=num_questions)
        if self.model_args.num_question_chunks > 1:
            idx = self.model_args.question_chunk_idx
            total = self.model_args.num_question_chunks

            print(f"chunk index: {idx}")
            assert 0 <= idx < total, f"Invalid chunk index: {idx}"

            total_questions = len(self.questions)
            chunk_size = total_questions // total
            start = idx * chunk_size
            end = total_questions if idx == total - 1 else (start + chunk_size)

            self.questions = self.questions[start:end]
        self.question_type_list = self.get_question_type_list()
        self.vlm = VLMWrapper(model_name=self.model_args.vlm_model_name, qa_model_name=self.model_args.vlm_qa_model_name)

        # ================== NEW OR MODIFIED BEGIN ==================
        if os.path.exists(os.path.join(self.model_args.output_dir, f"results.json")):
            with open(os.path.join(self.model_args.output_dir, f"results.json"), 'r') as f:
                self.results = json.load(f)
        else:
            self.results = {
                "current": None,
                "parsing_err_stats":{
                    "scores": 0,
                    "answer": 0,
                    "answer_qid": [],
                    "scores_qid": [],
                },
                "accuracy": {
                    "all": None,
                    "types" : {
                        question_type: None for question_type in self.question_type_list
                    }
                },
                "skip_indices": [],
                "progress":{
                    question_type: {
                        "correct": [],
                        "wrong": [],
                    } for question_type in self.question_type_list
                
                }
            }
            # self.save_results()

    def get_question_type_list(self):
        """
        Scan the question type and return the corresponding type.
        """
        question_type_list = []
        for question in self.questions:
            question_type = question["question_type"]
            if question_type not in question_type_list:
                question_type_list.append(question_type)
        return question_type_list
    def _process_bbox(self, response: str):
        """Parse LLM response to a bounding box: [(x1,y1), (x2,y2)]. response is in format (150,160):(180,220)."""
        try:
            if "Output:" in response:
                response = response.split("Output:")[1]
            list_= []
            coordinates = response.split(":")
            for coordinate in coordinates:
                coordinate = coordinate.strip()[1:-1].split(',')
                list_.append((int(coordinate[0]), int(coordinate[1])))
            return list_
        except Exception:
            pass
        return "out of control"
    def save_results(self):
        """Save results to JSON file."""
        with open(os.path.join(self.model_args.output_dir, f"results.json"), 'w') as f:
            json.dump(self.results, f, indent=4)
        # ================== NEW OR MODIFIED END ==================

    def run(self):

        for question in tqdm(self.questions):
            # ------------------------------------------------------------------
            #  Quick filters & deduplication
            # ------------------------------------------------------------------
            if question["question_type"] in ["other"]:
                self.results["skip_indices"].append(question["database_idx"])
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
            for step in range(self.model_args.max_steps_per_question):
                print(f"\n[SpatialVQA] ===== Step {step} for QID {qid} =====")

                for _ in range(self.model_args.max_tries_gpt):
                    sys_prompt, content = self.vlm.format_prompt(prompt_type="answer_baseline",
                        question=question["question"],
                        answer_choices=question["answer_choices"],
                        images=[primary_img_path, helper_img_path] if len(question['img_paths']) > 1 else [primary_img_path],
                    )
                    response = self.vlm.run_prompt("answer_baseline", sys_prompt, content)
                    print("[LLM]", response)
                    result = self._process_answer(response, question)
                    if result != "out of control":
                        break

                self._dump_llm_interaction(save_dir, step, question, response, result, None, None)
                if result == "out of control":
                    result = "wrong"

                if result in ("correct", "wrong"):
                    self.results["progress"][question["question_type"]][result].append(qid)
                    break

                print(f"===============End of iteration {step}================")

            print("result:", result)
            
            # ----------------------------------------------------------
            #  Terminal answer?  –> store + break
            # ----------------------------------------------------------

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
            self.save_results()

    def load_skips(self, skip_file):
        if os.path.exists(skip_file):
            with open(skip_file, 'r') as f:
                return json.load(f)
        else:
            return []


    def save_skips(self):
        with open(self.skip_file, 'w') as f:
            json.dump(self.skip_indices, f, indent=4)

    def calculate_types_count(results):
        count_map = {
            'correct': [],
            'wrong': [],
        }
        for result in results:
            if result in count_map.keys():
                count_map[result] += 1
        return count_map

    def select_questions(self, input_file, seed=None, num_questions=1):
        """
        Select a specified number of questions from a JSON file using a custom seed.

        :param input_file: Path to the JSON file containing a list of questions
        :param seed: Custom seed for random number generation (int or None)
        :param num_questions: Number of questions to select
        :return: List of selected questions
        """
        with open(input_file, 'r') as f:
            all_questions = json.load(f)

        # Set the random seed (if None, it won't fix the seed)
        random.seed(seed)

        # Ensure we don't request more questions than are available
        num_to_select = min(num_questions, len(all_questions))

        # Use random.sample to select multiple distinct items
        selected = random.sample(all_questions, k=num_to_select)
        return selected

    def init_prompt(self, question):
        self.chat_api.message = [
            {"role": "system", "content": SYS},
        ]
        new_prompt = self.prompt.format(
                question=question['question'], 
                answer_choice=question['answer_choices']
            )
        self.chat_api.add_user_image_message(
            question["img_paths"], 
            new_prompt
        )
        return new_prompt
    def _dump_llm_interaction(self, save_dir, step, question, response, result, actions, magnitude):
        prompt = copy.deepcopy(self.vlm.curr_prompt)
        log = {
            "question": question,
            "result": result,
            "action_list": actions,
            "magnitude": magnitude,
            "llm_response": response,
            "prompt": prompt,
        }
        step_dir = os.path.join(save_dir, f"step_{step}")
        os.makedirs(step_dir, exist_ok=True)
        with open(os.path.join(step_dir, "gpt.json"), "w") as f:
            json.dump(log, f, indent=2)

    def _process_answer(self, response: str, question: dict, fwd=0.075, turn=3):
        """Parse LLM response and map to (result, actions, magnitude)."""
        response_l = response.lower()
        try:
            if any(c.lower() in response_l for c in question["answer_choices"]):
                return "correct" if question["correct_answer"].lower() in response_l else "wrong"
        except Exception:
            pass
        return "out of control"

if __name__ == "__main__":
    pipeline = PipelineBase()
    pipeline.run()