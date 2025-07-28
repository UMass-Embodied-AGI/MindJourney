import base64
import copy
from PIL import Image
from utils.InternVL3 import *

SYS = """
You are an AI assistant designed to help us understand spatial relationship in 3D indoor scene and finish visual question answering.
"""

BASELINE_PROMPT = """
You will be given one or two images and a spatial reasoning reasoning questions.
Your goal is to answer the spatial related question correctly.

Directly output an answer from the answer choices provided below.
You can add some analysis in your response, but remember to format the end of your answer according to the rule.

Now, according to the following image, answer the question from provided choices:
Question: {question}
Answer Choice: {answer_choice}

Answer: 
"""


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def format_gpt_content(contents):
    formatted_content = []
    for c in contents:
        formatted_content.append({"type": "text", "text": c[0]})
        if len(c) == 2: # has image
            formatted_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(c[1])}",
                        "detail": "high",
                    },
                }
            )
    return formatted_content

def format_internvl3_content(contents, model_device=None):
    formatted_content = {
        "question": "",
        "num_patches_list": [],
    }
    pixel_values = []
    for c in contents:
        formatted_content["question"] += c[0]
        formatted_content["question"] += "\n"
        if len(c) == 2: # has image
            formatted_content["question"] += "<image>\n"
            img_tensor = load_image(c[1], max_num=12).to(torch.bfloat16)
            if model_device is not None:
                img_tensor = img_tensor.to(model_device)
            else:
                img_tensor = img_tensor.cuda()

            pixel_values.append(img_tensor)
            formatted_content["num_patches_list"].append(img_tensor.size(0))
    formatted_content["pixel_values"] = torch.cat(pixel_values, dim=0)
    return formatted_content

def format_spatial_vqa_prompt_answer_baseline(
    question: str,
    answer_choices: list,
    images: list,
) -> (str, list):
    """
    Format a ChatGPT prompt (with optional images) for a spatial VQA scenario.
    
    Args:
        question (str): The question to answer.
        answer_choices (list): The list of possible answer choices.
        images (list): A list of local file paths to images for the current view.
        
    Returns:
        (str, list):
            - A system prompt describing ChatGPT's overarching role & guidelines.
            - A list of pieces of content (text or (text, image)) for ChatGPT.
            The 'image' part is a Base64-encoded string.
    """
    
    # 1) System prompt describing the assistant’s overall role & rules
    sys_prompt = (
        "You are an AI assistant designed to help with spatial reasoning in a 3D indoor scene. "
        "You must analyze any provided images or observations and answer the question.\n\n"
    )
    
    # 2) Build the content list: each element is text or (text, base64_image).
    content = []
    
    # a) Intro: mention current images (if any)
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        content.append((f"\nImage 1 is your current egocentric view\n",))
    else:
        content.append(("No image provided.\n\n",))
    
    # b) Present the question and answer choices
    q_text = f"Question: {question}\n"
    ac_text = "Answer Choices:\n"
    for choice in answer_choices:
        ac_text += f"{choice}\n"
    content.append((q_text,))
    content.append((ac_text,))
    
    # e) Final instructions and the "Answer:" line
    instructions = (
        "Output the exact answer from the choices.\n"
        "Answer: "
    )
    content.append((instructions,))
    
    return sys_prompt, content

def format_spatial_vqa_prompt_answer_scaling(
    question: str,
    answer_choices: list,
    images: list,
    action_consequences: dict
) -> (str, list):
    """
    Format a ChatGPT prompt for a spatial VQA scenario in which we
    present multiple candidate actions *before* the assistant chooses one.
    
    Arguments:
        question (str): The question to answer.
        answer_choices (list): The list of possible answer choices.
        images (list): A list of local file paths to the current/initial view images.
        action_consequences (dict): A nested dictionary of candidate actions and their corresponding images.
            The structure is:
                {
                    "action_1": {
                        "subaction_1": "path_to_image",
                        "subaction_2": "path_to_image",
                        ...
                    },
                    "action_2": {
                        "subaction_1": "path_to_image",
                        "subaction_2": "path_to_image",
                        ...
                    },
                    ...
                }

    Returns:
        (str, list):
            - A system prompt describing ChatGPT's role & guidelines.
            - A list of pieces of content (text or (text, base64_image)) for ChatGPT.
              The 'image' part is a Base64-encoded string.
    """
    
    # ------------------ 1) System Prompt ------------------
    sys_prompt = (
        "You are an AI assistant designed to help with spatial reasoning in a 3D indoor scene. "
        "You must analyze any provided images or observations and answer the question.\n\n"
        "Rules:\n"
        "1. You should output the exact answer from the choices.\n"
        "2. You will be provided with multiple imagined views if you taking corresponding actions to help you answer the questions.\n"
        "3. Your final line must only include the exact answer choice.\n"
    )
    
    # Prepare the content list: text or (text, base64_image)
    content = []
    
    # ------------------ 2) Current Images ------------------
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        content.append(("\nImage 1 is your current egocentric view\n",))
    else:
        content.append(("No image provided.\n\n",))
    
    # ------------------ 3) The Question & Choices ------------------
    q_text = f"Question: {question}\n\n"
    ac_text = "Answer Choices:\n"
    for choice in answer_choices:
        ac_text += f"  - {choice}\n"
    ac_text += "\n"
    
    content.append((q_text,))
    content.append((ac_text,))
    
    # ------------------ 4) Present Candidate Actions + Images ------------------
    # e.g. "turn-left 30" -> [3 images], "turn-right 30" -> [3 images], etc.
    actions_intro = (
        "Below are the imagined views you would obtain if you took the corresponding actions. "
        "These are provided to help you answer the question.\n"
        "You can include them in your reasoning, but you should still only output the exact answer at the last line\n"
    )
    content.append((actions_intro,))
    
    for action_str, subaction_consequences in action_consequences.items():
        content.append((f"Action: {action_str}\n",))
        for subaction_str, img_path in subaction_consequences.items():
            content.append((f"{subaction_str}\n", img_path))
        content.append(("\n",))
    
    # ------------------ 5) Final Instructions + "Answer:" Prompt ------------------
    # The user can either pick an answer from the list or pick an action from the action_consequences.
    instructions = (
        "Output the exact answer from the choices.\n"
        "Answer: "
    )
    content.append((instructions,))
    
    return sys_prompt, content

def format_spatial_vqa_prompt_scores(
    question: str,
    answer_choices: list,
    images: list,
    action_consequences: list,
    sys_prompt: str,
) -> (str, list):
    
    """
    Score the imaginations during the beam search process.
    """
    
    # ------------------ 1) System Prompt ------------------
    
    # Prepare the content list: text or (text, base64_image)
    content = []
    
    # ------------------ 2) Current Images ------------------
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        content.append(("\nImage 1 is your current egocentric view\n",))
    else:
        content.append(("No image provided.\n\n",))
    
    # ------------------ 3) The Question & Choices ------------------
    q_text = f"Question: {question}\n\n"
    ac_text = "Answer Choices:\n"
    for choice in answer_choices:
        ac_text += f"  - {choice}\n"
    ac_text += "\n"
    
    content.append((q_text,))
    content.append((ac_text,))
    
    # ------------------ 4) Present Candidate Actions + Images ------------------
    # e.g. "turn-left 30" -> [3 images], "turn-right 30" -> [3 images], etc.

    action_intro = (
        f"Below are the imagined views after taking actions."
    )
    for index, action_consequence in enumerate(action_consequences):
        action_str, subaction_consequence, img_path = action_consequence
        content.append((action_intro,))
        content.append((f"Imagined image of index {str(index)} if you {subaction_consequence}:\n", img_path))
        content.append(("\n",))
    
    # ------------------ 5) Final Instructions + "Answer:" Prompt ------------------
    # The user can either pick an answer from the list or pick an action from the action_consequences.
    instructions = (
        "Output a list of scores.\n"
        "Output: "
    )
    content.append((instructions,))
    
    return sys_prompt, content

def format_spatial_vqa_prompt_answer_baseline_fill_in_blank(
    question: str,
    answer_choices: list,
    images: list = None,
) -> (str, list):
    """
    Format a ChatGPT prompt (with optional images) for a spatial VQA scenario.
    
    Args:
        question (str): The question to answer.
        answer_choices (list): The list of possible answer choices.
        images (list): A list of local file paths to images for the current view.
        
    Returns:
        (str, list):
            - A system prompt describing ChatGPT's overarching role & guidelines.
            - A list of pieces of content (text or (text, image)) for ChatGPT.
              The 'image' part is a Base64-encoded string.
    """
    
    # 1) System prompt describing the assistant’s overall role & rules
    sys_prompt = (
        "You are an AI assistant designed to help with spatial reasoning in a 3D indoor scene. "
        "You must analyze any provided images or observations and answer the question.\n\n"
        "Rules:\n"
        "1. You should output the exact answer to fill in the blank, like directly output a floating-point number.\n"
        "2. Your final line must only include the exact answer choice.\n"
        "3. If there is an example format in the question, you should strictly follow it, otherwise you should only output a float-point number as the exact answer.\n"
        r"4. The final answer MUST BE put in \boxed{}."
    )
    
    # 2) Build the content list: each element is text or (text, base64_image).
    content = []
    
    # a) Intro: mention current images (if any)
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        content.append(("\n",))
    else:
        content.append(("No image provided.\n\n",))
    
    # b) Present the question and answer choices
    q_text = f"Question: {question}\n"
    # ac_text = "Answer Choices:\n"
    # for choice in answer_choices:
    #     ac_text += f"{choice}\n"
    content.append((q_text,))
    # content.append((ac_text,))
    
    # e) Final instructions and the "Answer:" line
    instructions = (
        "Output the exact answer in a float-point number format.\n"
        "Answer: "
    )
    content.append((instructions,))
    
    return sys_prompt, content

def format_spatial_vqa_prompt_scores_fill_in_blank(
    # Currently hard code n=2
    question: str,
    answer_choices: list,
    images: list,
    action_consequences: list,
    sys_prompt: str,
) -> (str, list):
    
    """
    Score the views during the beam search process.
    """
    
    # ------------------ 1) System Prompt ------------------
    
    # Prepare the content list: text or (text, base64_image)
    content = []
    
    # ------------------ 2) Current Images ------------------
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        content.append(("\nImage 1 is your current egocentric view\n",))
    else:
        content.append(("No image provided.\n\n",))
    
    # ------------------ 3) The Question & Choices ------------------
    q_text = f"Question: {question}\n\n"
    # ac_text = "Answer Choices:\n"
    # for choice in answer_choices:
    #     ac_text += f"  - {choice}\n"
    # ac_text += "\n"
    
    content.append((q_text,))
    # content.append((ac_text,))
    
    # ------------------ 4) Present Candidate Actions + Images ------------------
    # e.g. "turn-left 30" -> [3 images], "turn-right 30" -> [3 images], etc.

    action_intro = (
        f"Below are the imagined views after taking actions."
    )
    for index, action_consequence in enumerate(action_consequences):
        action_str, subaction_consequence, img_path = action_consequence
        content.append((action_intro,))
        content.append((f"Imagined image of index {str(index)} if you {subaction_consequence}:\n", img_path))
        content.append(("\n",))
    
    # ------------------ 5) Final Instructions + "Answer:" Prompt ------------------
    # The user can either pick an answer from the list or pick an action from the action_consequences.
    instructions = (
        "Output a list of scores.\n"
        "Output: "
    )
    content.append((instructions,))
    
    return sys_prompt, content

def format_spatial_vqa_prompt_rank(
    # Currently hard code n=2
    question: str,
    answer_choices: list,
    images: list,
    action_consequences: list,
) -> (str, list):
    
    """
    Rank the views during the beam search process.
    """
    
    # ------------------ 1) System Prompt ------------------
    sys_prompt = (
        "You are an AI assistant designed to help with spatial reasoning in a 3D indoor scene. "
        "You must analyze any provided images and rank indexes of imagined images from most relevant to least relevant.\n\n"
        "Rules:\n"
        "1. You'll be provided with images (including imagined images), a question, and a set of answer choices. You should rank most relevant images that can help you answer the question from the choices.\n"
        "2. You should output a list of indexes, separated by ','. For example: Output: 3,1,2,0\n"
    )
    
    # Prepare the content list: text or (text, base64_image)
    content = []
    
    # ------------------ 2) Current Images ------------------
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    
    if images:
        for idx, img_path in enumerate(images):
            encoded_img = encode_image(img_path)
            content.append((f"Image {idx + 1}:", encoded_img))
        content.append(("\nImage 1 is your current egocentric view\n",))
    else:
        content.append(("No image provided.\n\n",))
    
    # ------------------ 3) The Question & Choices ------------------
    q_text = f"Question: {question}\n\n"
    ac_text = "Answer Choices:\n"
    for choice in answer_choices:
        ac_text += f"  - {choice}\n"
    ac_text += "\n"
    
    content.append((q_text,))
    content.append((ac_text,))
    
    # ------------------ 4) Present Candidate Actions + Images ------------------
    # e.g. "turn-left 30" -> [3 images], "turn-right 30" -> [3 images], etc.

    action_intro = (
        f"Below are the imagined views after taking actions."
    )
    for index, action_consequence in enumerate(action_consequences):
        action_str, subaction_consequence, img_path = action_consequence
        content.append((action_intro,))
        encoded_img = encode_image(img_path)
        content.append((f"Imagined image of index {str(index)} if you {subaction_consequence}:\n", encoded_img))
        content.append(("\n",))
    
    # ------------------ 5) Final Instructions + "Answer:" Prompt ------------------
    # The user can either pick an answer from the list or pick an action from the action_consequences.
    instructions = (
        "Output a list of indexes from most relevant image to least relevant image.\n"
        "Output: "
    )
    content.append((instructions,))
    
    return sys_prompt, content


def format_spatial_vqa_prompt_answer_scaling_fill_in_blank(
    question: str,
    answer_choices: list,
    images: list,
    action_consequences: dict
) -> (str, list):
    """
    Format a ChatGPT prompt for a spatial VQA scenario in which we
    present multiple candidate actions *before* the assistant chooses one.
    
    Arguments:
        question (str): The question to answer.
        answer_choices (list): The list of possible answer choices.
        images (list): A list of local file paths to the current/initial view images.
        action_consequences (dict): A nested dictionary of candidate actions and their corresponding images.
            The structure is:
                {
                    "action_1": {
                        "subaction_1": "path_to_image",
                        "subaction_2": "path_to_image",
                        ...
                    },
                    "action_2": {
                        "subaction_1": "path_to_image",
                        "subaction_2": "path_to_image",
                        ...
                    },
                    ...
                }

    Returns:
        (str, list):
            - A system prompt describing ChatGPT's role & guidelines.
            - A list of pieces of content (text or (text, base64_image)) for ChatGPT.
              The 'image' part is a Base64-encoded string.
    """
    
    # ------------------ 1) System Prompt ------------------
    sys_prompt = (
        "You are an AI assistant designed to help with spatial reasoning in a 3D indoor scene. "
        "You must analyze any provided images or observations and answer the question.\n\n"
        "Rules:\n"
        "1. You should output the exact answer to fill in the blank, like directly output a floating-point number.\n"
        "2. You will be provided with multiple imagined views if you taking corresponding actions to help you answer the questions.\n"
        "3. You can include minimal reasoning, but your final line must only include the exact answer.\n"
        "4. If there is an example format in the question, you should strictly follow it, otherwise you should only output a float-point number as the exact answer.\n"
        r"5. The final answer MUST BE put in \boxed{}."
    )
    
    # Prepare the content list: text or (text, base64_image)
    content = []
    
    # ------------------ 2) Current Images ------------------
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        content.append(("\nImage 1 is your current egocentric view\n",))
    else:
        content.append(("No image provided.\n\n",))
    
    # ------------------ 3) The Question & Choices ------------------
    q_text = f"Question: {question}\n\n"
    # ac_text = "Answer Choices:\n"
    # for choice in answer_choices:
    #     ac_text += f"  - {choice}\n"
    # ac_text += "\n"
    
    content.append((q_text,))
    # content.append((ac_text,))
    # content.append((ac_text,))
    
    # ------------------ 4) Present Candidate Actions + Images ------------------
    # e.g. "turn-left 30" -> [3 images], "turn-right 30" -> [3 images], etc.
    actions_intro = (
        "Below are the imagined views you would obtain if you took the corresponding actions.\n"
        "If there are more than one image in the question, these imaged views are based on the first image.\n"
        "These are provided to help you answer the question.\n"
        "You can include them in your reasoning, but you should still only output the exact answer at the last line\n"
    )
    content.append((actions_intro,))
    
    for action_str, subaction_consequences in action_consequences.items():
        content.append((f"Action: {action_str}\n",))
        for subaction_str, img_path in subaction_consequences.items():
            content.append((f"{subaction_str}\n", img_path))
        content.append(("\n",))
    
    # ------------------ 5) Final Instructions + "Answer:" Prompt ------------------
    # The user can either pick an answer from the list or pick an action from the action_consequences.
    instructions = (
        "Output the exact answer from the question.\n"
        "Answer: "
    )
    content.append((instructions,))
    
    return sys_prompt, content

def format_spatial_vqa_prompt_bbox(
    question: str,
    answer_choices: list,
    images: list,
) -> (str, list):
    sys_prompt = (
        "You are an AI assistant designed to help with spatial reasoning in a 3D indoor scene. "
        "You must analyze the image and answer the question.\n\n"
        "Rules:\n"
        "1. Output the bounding box in your current egocentric view of the area most important and relevant for answering the question. For those questions containing marks, it is important to have the bounding box include the object that marked with the number mentioned in the question.\n"
        "2. The output should only contain two integer coordinates of the top-left and bottom-right corners of the bounding box, separated by ':' in the format (x1,y1):(x2,y2).\n"
        "3. Only output None if you are very uncertain about the bounding box location or it is not necessary for answering the question. This case is rare to happen.\n"
    )
    content = []
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        content.append((f"\nImage 1 is your current egocentric view of size {Image.open(images[0]).size}\n",))
    else:
        content.append(("No image provided.\n\n",))
    q_text = f"Question: {question}\n\n"
    ac_text = "Answer Choices:\n"
    for choice in answer_choices:
        ac_text += f"  - {choice}\n"
    ac_text += "\n"
    content.append((q_text,))
    content.append((ac_text,))
    instructions = (
        "Output either the bounding box coordinates in the format (x1,y1):(x2,y2) or None if uncertain or not needed.\n"
        "Output: "
    )
    content.append((instructions,))
    return sys_prompt, content