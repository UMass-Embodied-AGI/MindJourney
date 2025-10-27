export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="disabled"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export PYTHONPATH=$PYTHONPATH:./
export TOKENIZERS_PARALLELISM=true
GPU_IDS="0"

echo "Running WAN2.2 pipeline script"

chunk_id=$1
num_questions=150
scaling_strategy="beam_search_double_rank"
question_type="None"

vlm_model_name="gpt-4o"
vlm_qa_model_name="gpt-4o"

helpful_score_threshold=8
exploration_score_threshold=8
max_images=2
max_steps=2
num_beams=2
inference_step=20

checkpoint_step=92000
num_top_candidates=18

export WORLD_MODEL_TYPE="wan2.2"
export QUESTION_DATASET_TYPE="SAT_test"
export WAN_CKPT_PATH="Your Checkpoint Path"

dataset_type="SAT_test"
input_dir="Your Data Path"

output_dir="results_wan2.2/${checkpoint_step}/${vlm_model_name}_${dataset_type}_${num_questions}_${max_steps}_${num_beams}_${helpful_score_threshold}_${exploration_score_threshold}_${inference_step}_${num_top_candidates}"

num_question_chunks=5
chunk_indices=($chunk_id)
echo "chunks_indices"

for idx in "${chunk_indices[@]}"; do
  cmd="python pipelines/pipeline_wan_scaling_beam_search_double_rank.py \
    --vlm_model_name $vlm_model_name \
    --vlm_qa_model_name $vlm_qa_model_name \
    --num_questions $num_questions \
    --output_dir $output_dir \
    --input_dir $input_dir \
    --scaling_strategy $scaling_strategy \
    --question_type $question_type \
    --helpful_score_threshold $helpful_score_threshold \
    --exploration_score_threshold $exploration_score_threshold \
    --max_images $max_images \
    --sampling_interval_angle 9 \
    --sampling_interval_meter 0.25 \
    --fixed_rotation_magnitudes 27 \
    --fixed_forward_magnitudes 0.75 \
    --max_steps_per_question $max_steps \
    --num_top_candidates $num_top_candidates \
    --num_beams $num_beams \
    --max_tries_gpt 5 \
    --num_frames 25 \
    --frame_interval 1 \
    --max_inference_batch_size 4 \
    --split test \
    --num_question_chunks $num_question_chunks \
    --question_chunk_idx $idx \
    "
  echo "Launching chunk $idx:"
  echo "Running command: $cmd"
  eval "$cmd" &
  # eval "$cmd"
done
wait


