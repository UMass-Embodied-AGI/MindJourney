export PYTHONPATH=$PYTHONPATH:./
# Launch experiments with different hyperparameters

chunk_id=$1

num_questions=150
scaling_strategy="beam_search_double_rank_multiframe"
question_type="None"

vlm_model_name="gpt-4o"
vlm_qa_model_name=None # pass None will be interpreted as "None" anyway; None means qa_vlm_model_name is same as vlm_model_name

helpful_score_threshold=8
exploration_score_threshold=8
max_images=2
max_steps=3

export WORLD_MODEL_TYPE="svc" # choose from "cogvideox" or "svc"
export QUESTION_DATASET_TYPE="SAT_test" # choose from "SAT_val" , "SAT_test" , "3DSRBench", "spar"
dataset_type="SAT_test"
input_dir="data"
split="test"

output_dir="results/results_svc_${vlm_model_name}_${dataset_type}_${num_questions}_${max_steps}_${exploration_score_threshold}_${helpful_score_threshold}_${max_images}"
export NUM_OF_FRAMES=20

num_question_chunks=5             # <<< split into 5 chunks
chunk_indices=($chunk_id)             # list of indices

for idx in "${chunk_indices[@]}"; do
  # Optional: pick a free GPU per process.
  # export CUDA_VISIBLE_DEVICES=$idx     # uncomment if you have ≥3 GPUs
  cmd="python pipelines/pipeline_svc_scaling_spatial_beam_search.py \
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
    --sampling_interval_angle 3 \
    --sampling_interval_meter 0.25 \
    --fixed_rotation_magnitudes 27 \
    --fixed_forward_magnitudes 0.75 \
    --max_steps_per_question $max_steps \
    --num_top_candidates 6 \
    --num_beams 2 \
    --max_tries_gpt 2 \
    --num_frames $((NUM_OF_FRAMES+1)) \
    --frame_interval 3 \
    --max_inference_batch_size 1 \
    --split $split \
    --num_question_chunks $num_question_chunks \
    --question_chunk_idx $idx \
    \
    --task img2trajvid_s-prob \
    --replace_or_include_input True \
    --cfg 4.0 \
    --guider 1 \
    --L_short 576 \
    --num_targets 8 \
    --use_traj_prior True \
    --chunk_strategy interp"

  echo "Launching chunk $idx:"
  echo "  $cmd"
  eval "$cmd" &                  # run in the background
  # eval "$cmd"                  # run in the front
done
wait

#--enable_slicing \
#--enable_tiling \
# --resume_from_checkpoint latest
