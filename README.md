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

We propose **MindJourney**, a test-time scaling framework that utilizes the 3D imagination capability of World Models to improve VLMs' Spatial Reasoning ability.

![](assets/teaser.png)

---

## News

- This code release is a work in progress - we'll update it with more details soon.
- [2025/07] Inference code for SAT with Stable-Virtual-Camera is released.
- [2025/07] [Paper](https://www.arxiv.org/abs/todo) is on arXiv.


## Installation

Set up the environment by running the following command:

```bash
conda create -n mindjourney python=3.10
pip install -e stable_virtual_camera/
pip install -r requirements.txt
```

Prepare the Azure OpenAI API endpoint in api.py by replacing "YOUR_API_ENDPOINT":

```python
self.azure_endpoint = "YOUR_API_ENDPOINT"
```

Export the Azure OpenAI API key:

```bash
export AZURE_OPENAI_API_KEY=YOUR_API_KEY
```

## Set Up Hugging Face Access to the Stable Virtual Camera model weights

Go to: https://huggingface.co/stabilityai/stable-virtual-camera  
Log in and request access

Then log in via CLI (using HF token):

```bash
huggingface-cli login
```

## Process SAT Dataset

```bash
python utils/data_process.py --split val
python utils/data_process.py --split test
```

## Run MindJourney on SAT Dataset

### Baseline
```bash
bash scripts/pipeline_baseline.sh
```

### Stable-Virtual-Camera
```bash
bash scripts/pipeline_svc_SAT_scaling_spatial_beam_search.sh
```
