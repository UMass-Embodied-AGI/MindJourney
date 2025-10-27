<br/>
  <div align="center">
    <img src="assets/MindJourney_logo_transparent.png" alt="MindJourney Logo" height="48" style="vertical-align:middle">
  </div>
  <h1 align="center" style="font-size: 1.7rem">Test-Time Scaling with World Models for Spatial Reasoning</h1>
  <p align="center">
    <a href="https://yyuncong.github.io/">Yuncong Yang</a>,
    <a href="https://jiagengliu02.github.io/">Jiageng Liu</a>,
    <a href="https://cozheyuanzhangde.github.io/">Zheyuan Zhang</a>,
    <a href="https://rainbow979.github.io/">Siyuan Zhou</a>,
    <a href="https://cs-people.bu.edu/rxtan/">Reuben Tan</a>,
    <a href="https://jwyang.github.io/">Jianwei Yang</a>,
    <a href="https://yilundu.github.io/">Yilun Du</a>,
    <a href="https://people.csail.mit.edu/ganchuang">Chuang Gan</a>
  </p>
  <p align="center">
    <a href="https://arxiv.org/abs/2507.12508">
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://umass-embodied-agi.github.io/MindJourney/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
  </p>
</p>

---

## Introduction

MindJourney is a test-time scaling framework that leverages the 3D imagination capability of World Models to strengthen spatial reasoning in Vision-Language Models (VLMs). We evaluate on the SAT dataset and provide both a baseline pipeline and a Stable Virtual Camera (SVC) based spatial beam search pipeline.

![](assets/teaser.png)

---

## News

- Actively maintained: baseline and SVC spatial beam-search pipelines are available with unified arguments and result logging.
- 2025/07: Inference code for SAT with Stable Virtual Camera released.
- 2025/07: Paper is on arXiv: https://arxiv.org/abs/2507.12508

---

## Repository Structure

- `pipelines/`
  - `pipeline_baseline.py`: baseline inference without a world model.
  - `pipeline_svc_scaling_spatial_beam_search.py`: SVC-based spatial beam search.
- `scripts/`
  - `pipeline_baseline.sh`: baseline example script.
  - `inference_pipeline_svc_scaling_parallel_sat_test.sh`: alternative SVC inference driver.
  - `inference_pipeline_wan_scaling_parallel_sat-test.sh`: example driver for WAN-based experiments.
- `utils/`
  - `api.py`: Azure OpenAI wrapper and configuration.
  - `args.py`: unified CLI arguments (pipeline + SVC).
  - `vlm_wrapper.py`, `prompt_formatting.py`: VLM wrapper and prompt construction.
  - `data_process.py`: SAT dataset preprocessing (download and JSON organization).
- `stable_virtual_camera/`: SVC module (editable install; see `pyproject.toml`).
- `assets/`: logo and figures.
- `wan2.2/`: WAN-related experimental code.

---

## Environment and Dependencies

We recommend two Conda environments to isolate the main runtime and SVC.

1) Main runtime (VLM + framework):

```bash
conda create -n mindjourney python=3.11 -y
conda activate mindjourney

# CUDA 12.6 builds of PyTorch (adjust if needed)
pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 \
  --extra-index-url https://download.pytorch.org/whl/cu126

# General dependencies
pip install -r requirements.txt
```

2) Stable Virtual Camera (separate env to avoid conflicts):

```bash
conda create -n mindjourney_svc python=3.10 -y
conda activate mindjourney_svc

# Editable install of the SVC module (dependencies defined in pyproject.toml)
pip install -e stable_virtual_camera/

# Optionally reuse shared utilities if needed
pip install -r requirements_svc.txt
```

Hardware suggestions:
- NVIDIA GPU (24 GB VRAM or more recommended), CUDA 12.6 drivers
- Sufficient disk space for intermediate videos and results

---

## Configure Azure OpenAI (for GPT-family VLMs)

1) Set your Azure endpoint in `utils/api.py`:
- File: `MindJourney-dev-new/utils/api.py`
- Field: `AzureConfig.azure_endpoint = "YOUR_API_ENDPOINT"`

2) Export the API key:

```bash
export AZURE_OPENAI_API_KEY=YOUR_API_KEY
```

Supported models: `gpt-4o`, `gpt-4.1`, `o4-mini`, `o1`. You can also choose `OpenGVLab/InternVL3-8B` or `OpenGVLab/InternVL3-14B` (ensure adequate VRAM and dependencies).

---

## Stable Virtual Camera Access (Hugging Face)

Request access to SVC weights and login:

```bash
# https://huggingface.co/stabilityai/stable-virtual-camera
huggingface-cli login
```

---

## Data Preparation (SAT)

Prepare SAT from Hugging Face using the helper script:

```bash
python utils/data_process.py --split val
python utils/data_process.py --split test
```

Outputs under `./data/`:
- `val.json` / `test.json`: questions with choices, correct answers, and image paths
- Per-type splits: `val_<type>.json` / `test_<type>.json`
- Images: `./data/<split>/image_*.png`

Per-question JSON fields: `database_idx`, `question_type`, `question`, `answer_choices`, `correct_answer`, `img_paths`.

---

## Quickstart

Before running:
- Export `AZURE_OPENAI_API_KEY` and set `azure_endpoint` in `utils/api.py` if using GPT models
- Add repo root to `PYTHONPATH`: `export PYTHONPATH=$PYTHONPATH:./`
- Set `WORLD_MODEL_TYPE=svc` for Stable Virtual Camera

### Option A: Use scripts

- Baseline:

```bash
bash scripts/pipeline_baseline.sh
```

- SVC spatial beam search:

```bash
bash scripts/pipeline_svc_SAT_scaling_spatial_beam_search.sh
```

### Option B: Run with explicit arguments

- Baseline (no world model):

```bash
export WORLD_MODEL_TYPE="svc"
export PYTHONPATH=$PYTHONPATH:./

python pipelines/pipeline_baseline.py \
  --vlm_model_name gpt-4o \
  --vlm_qa_model_name None \
  --num_questions 150 \
  --output_dir results/results_baseline_gpt4o_test_150_2 \
  --input_dir data \
  --question_type None \
  --max_images 2 \
  --max_tries_gpt 5 \
  --split test \
  --num_question_chunks 1 \
  --question_chunk_idx 0
```

- Spatial beam search with SVC:

```bash
export WORLD_MODEL_TYPE="svc"
export PYTHONPATH=$PYTHONPATH:./

python pipelines/pipeline_svc_scaling_spatial_beam_search.py \
  --vlm_model_name gpt-4o \
  --vlm_qa_model_name None \
  --num_questions 150 \
  --output_dir results/svc_test_gpt4o_150_1_8_8_2 \
  --input_dir data \
  --scaling_strategy spatial_beam_search \
  --question_type None \
  --helpful_score_threshold 8 \
  --exploration_score_threshold 8 \
  --max_images 2 \
  --sampling_interval_angle 9 \
  --sampling_interval_meter 0.25 \
  --fixed_rotation_magnitudes 27 \
  --fixed_forward_magnitudes 0.75 \
  --max_steps_per_question 1 \
  --num_top_candidates 6 \
  --num_beams 3 \
  --max_tries_gpt 4 \
  --num_frames 9 \
  --frame_interval 3 \
  --max_inference_batch_size 1 \
  --split test \
  --num_question_chunks 1 \
  --question_chunk_idx 0 \
  --task img2trajvid_s-prob \
  --replace_or_include_input True \
  --cfg 4.0 \
  --guider 1 \
  --L_short 576 \
  --num_targets 8 \
  --use_traj_prior True \
  --chunk_strategy interp
```

Key arguments (see `utils/args.py` for full list):
- `--vlm_model_name` / `--vlm_qa_model_name`: scoring and answering VLMs (Azure OpenAI or InternVL3)
- `--num_questions`, `--split`: number of questions and split (`val`/`test`)
- `--max_steps_per_question`: max iterations per question (beam search)
- `--num_beams`, `--num_top_candidates`: beam width and candidate count
- `--helpful_score_threshold`, `--exploration_score_threshold`: filtering thresholds
- `--max_images`: max images per question (typically 1–2)

---

## Results and Logs

Outputs under `--output_dir`:
- `results.json`: overall accuracy, per-type accuracy, skipped indices, parsing stats
- `/<qid>/step_0/`: starting image(s) and `gpt.json` logs per question
- For SVC: generated videos and camera trajectories per candidate (e.g., `episode.pkl`, `episode.json`)

---

## Citation

If you find this repository helpful, please cite:

```
@article{mindjourney2025,
  title   = {Test-Time Scaling with World Models for Spatial Reasoning},
  author  = {Yang, Yuncong and Liu, Jiageng and Zhang, Zheyuan and Zhou, Siyuan and Tan, Reuben and Yang, Jianwei and Du, Yilun and Gan, Chuang},
  journal = {arXiv preprint arXiv:2507.12508},
  year    = {2025}
}
```

---

## Notes

- This repository is under active development; interfaces and scripts may change.
- Configure a valid Azure endpoint and API key if using GPT-family models; you are responsible for any API costs.
- SVC weights require approval on Hugging Face and appreciable VRAM.
- For issues with environments or arguments, see `utils/args.py` and code comments in `pipelines/`.
