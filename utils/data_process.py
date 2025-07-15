from datasets import load_dataset
from PIL import Image
import io
import argparse
import json
import os

def process(output_dir, max_per_type = None, split = 'val'):
    dataset = load_dataset("array/SAT", batch_size=128)
    selected_examples = []
    selected_examples_separate = {}

    all_question_types= []
    question_count = dict()
    
    os.makedirs(f"./data/{split}", exist_ok=True)
    resolutions = []
    for i in range(len(dataset[split])):
        print(i)
        example = dataset[split][i]
        question_type = example['question_type']

        if question_type not in all_question_types:
            all_question_types.append(str(question_type))
            question_count[str(question_type)] = 0
            selected_examples_separate[question_type] = []
        # print(all_question_types)
        if max_per_type != None:
            if question_count[question_type] >= max_per_type:
                continue
        question_count[question_type] += 1

        filename_list = []
        print(resolutions, all_question_types)
        for idx, im_bytes in enumerate(example['image_bytes']):
            # img = Image.open(io.BytesIO(im_bytes))
            img = im_bytes
            filename = f"./data/{split}/image_{i}_{idx}.png"
            filename_list.append(filename)
            resolution = img.size
            if resolution not in resolutions:
                resolutions.append(resolution)
            
            if split == 'test':
                width, height = img.size
                min_dim = min(width, height)
                left = (width - min_dim) // 2
                top = (height - min_dim) // 2
                right = left + min_dim
                bottom = top + min_dim
                img_cropped = img.crop((left, top, right, bottom))

                # Step 2: 512x512
                img = img_cropped.resize((512, 512), resample=Image.BICUBIC)
            img.save(filename, format="PNG")

        question = example['question']
        answer_choices = example['answers']
        correct_answer = example['correct_answer']
        question_data = {
                "database_idx":i,
                "question_type":question_type,
                "question":question,
                "answer_choices":answer_choices,
                "correct_answer":correct_answer,
                "img_paths": filename_list,
            }
        selected_examples.append(question_data)
        selected_examples_separate[question_type].append(question_data)

    with open(os.path.join(output_dir, f"{split}.json"), "w") as json_file:
        json.dump(selected_examples, json_file, indent=4)
    
    for q_type, examples in selected_examples_separate.items():
        json_file_path = os.path.join(output_dir, f"{split}_{q_type}.json")
        with open(json_file_path, "w") as json_file:
            json.dump(examples, json_file, indent=4)
        print(f"Saved {len(examples)} examples to {json_file_path}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='./data')
    parser.add_argument("--max_per_type", type=int, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    
    args = parser.parse_args()
    process(args.output_dir , args.max_per_type, args.split)