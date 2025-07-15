# A VLM wrapper for adapting both close-source (i.e. API) and open-source (i.e. local) models.
from utils.api import ChatAPI, AzureConfig
from utils.InternVL3 import *
from utils.prompt_formatting import *

class VLMWrapper:
    def __init__(self, model_name, qa_model_name=None):
        if model_name in ['gpt-4o', 'gpt-4.1', 'o4-mini', 'o1']:
            api_info = {
                "gpt-4o": {
                    "api_version": "2024-12-01-preview",
                    "api_price": 0.01,
                },
                "gpt-4.1": {
                    "api_version": "2024-12-01-preview",
                    "api_price": 0.005,
                },
                "o4-mini": {
                    "api_version": "2024-12-01-preview",
                    "api_price": 0.000375,
                },
                "o1": {
                    "api_version": "2024-12-01-preview",
                    "api_price": 0.005,
                }
            }
            # Assume we use the Azure OpenAI API
            config = AzureConfig(model_name, api_info[model_name]["api_version"], api_info[model_name]["api_price"]) # already set greedy decoding in the config
            self.model = ChatAPI(config)
            self.qa_model = None
            if qa_model_name not in (None, "None") and qa_model_name != model_name:
                qa_config = AzureConfig(qa_model_name, api_info[qa_model_name]["api_version"], api_info[qa_model_name]["api_price"])
                self.qa_model = ChatAPI(qa_config)
            self.prompt_style = 'gpt'
        elif model_name in ['OpenGVLab/InternVL3-8B', 'OpenGVLab/InternVL3-14B']:
            assert qa_model_name in (None, "None") or qa_model_name == model_name, "Separate Score/QA model is not supported for InternVL3."
            # device_map = split_model(model_name)
            device_map = "cuda:1"
            device_map = split_model(model_name)
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                load_in_8bit=False,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
                device_map=device_map).eval()
            # self.model = AutoModel.from_pretrained(
            #     model_name,
            #     torch_dtype=torch.bfloat16,
            #     load_in_8bit=False,
            #     low_cpu_mem_usage=True,
            #     use_flash_attn=True,
            #     trust_remote_code=True,
            #     device_map=device_map).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
            self.generation_config = dict(max_new_tokens=1024, do_sample=False) # greedy decoding
            self.prompt_style = 'internvl3'
        else:
            raise ValueError(f"Model {model_name} is not supported.")
        self.curr_prompt = None
    
    def format_prompt(self, prompt_type, question, answer_choices, images, action_consequences=None, sys_prompt=None):
        if prompt_type == 'bounding_box':
            return format_spatial_vqa_prompt_bbox(question=question, answer_choices=answer_choices, images=images)
        if prompt_type == 'answer_baseline':
            return format_spatial_vqa_prompt_answer_baseline(question=question, answer_choices=answer_choices, images=images)
        elif prompt_type == 'answer_scaling':
            return format_spatial_vqa_prompt_answer_scaling(question=question, answer_choices=answer_choices, images=images, action_consequences=action_consequences)
        elif prompt_type == "prompt_scores":
            return format_spatial_vqa_prompt_scores(question=question, answer_choices=answer_choices, images=images, action_consequences=action_consequences, sys_prompt=sys_prompt)
        elif prompt_type == "answer_baseline_fill":
            return format_spatial_vqa_prompt_answer_baseline_fill_in_blank(question=question, answer_choices=answer_choices, images=images)
        elif prompt_type == "answer_scaling_fill":
            return format_spatial_vqa_prompt_answer_scaling_fill_in_blank(question=question, answer_choices=answer_choices, images=images, action_consequences=action_consequences)
        elif prompt_type == "prompt_scores_fill":
            return format_spatial_vqa_prompt_scores_fill_in_blank(question=question, answer_choices=answer_choices, images=images, action_consequences=action_consequences, sys_prompt=sys_prompt)
        else:
            raise ValueError(f"Prompt type {prompt_type} is not supported.")
    
    def run_prompt(self, prompt_type, system_prompt, content):
        self.curr_prompt = {"system": system_prompt, "content": content}
        if self.prompt_style == 'gpt':
            content = format_gpt_content(content)
            if prompt_type[:7] == 'answer_' and self.qa_model is not None:
                response = self.qa_model.get_system_response_with_content(system_prompt, content)
            else:
                response = self.model.get_system_response_with_content(system_prompt, content)
            return response
        elif self.prompt_style == 'internvl3':
            content = format_internvl3_content(content, "cuda:1")
            # content = format_internvl3_content(content, None)
            # content = format_internvl3_content(content)
            question = content['question']
            pixel_values = content['pixel_values']
            num_patches_list = content['num_patches_list']
            response, history = self.model.chat(self.tokenizer, pixel_values, system_prompt+'\n'+question, self.generation_config, num_patches_list=num_patches_list, history=None, return_history=True)
            return response
        else:
            raise ValueError(f"Prompt style {self.prompt_style} is not supported.")