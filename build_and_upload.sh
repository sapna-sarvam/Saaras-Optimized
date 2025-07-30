#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

# Common parameters
model_name=${MODEL_NAME} #model_version in llm block
model_version=${MODEL_VERSION}
max_batch_size=${MAX_BATCH_SIZE:-96}
max_input_len=${MAX_INPUT_LEN:-4096}
max_seq_len=${MAX_SEQ_LEN:-4096}
base_storage_path=${BASE_STORAGE_PATH}
base_dir=${BASE_DIR:-"/models"}
is_asr=${ASR:-false}
tp_size=${TP_SIZE:-1}
org=${ORG:-"sarvam"}

# ASR-specific parameters
decoder_seq_len=${DECODER_SEQ_LEN:-114}
decoder_input_len=${DECODER_INPUT_LEN:-14}
asr_model_version=${ASR_MODEL_VERSION:-"large-v2"}

# LLM-specific parameters
quantize_fp8_flag=${QUANTIZE_FP8:-false}
base_model=${BASE_MODEL:-"qwen"}


if [[ "$is_asr" == "true" ]]; then

    # Directory structure for ASR
    hf_model_dir="$base_dir/base_models/${model_name}/${model_version}"
    distil_output_dir="$base_dir/base_models/${model_name}-converted/${model_version}"
    checkpoint_convert_dir="$base_dir/convert_models/${model_name}-${model_version}-trt-converted"
    trt_engine_dir="$base_dir/trt_engines/${model_name}-${model_version}-bs${max_batch_size}/${tp_size}-gpu"
    dtype="float16"

    echo "Downloading model from Hugging Face..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${org}/${model_name}',
    revision='${model_version}',
    local_dir='${hf_model_dir}',
)"

    echo "Building Whisper ASR model..."
    set -x

    # Step 1: Convert from distil whisper format
    echo "Converting from distil whisper format..."
    python /app/tensorrt_llm/examples/models/core/whisper/distil_whisper/convert_from_distil_whisper.py \
        --model_name "${org}/${model_name}" \
        --output_dir "$distil_output_dir" \
        --cache_dir "$hf_model_dir" \
        --output_name "$asr_model_version"

    # Step 2: Convert checkpoint for TensorRT-LLM
    echo "Converting checkpoint for TensorRT-LLM..."
    python /app/tensorrt_llm/examples/models/core/whisper/convert_checkpoint.py \
        --model_dir "$distil_output_dir" \
        --output_dir "$checkpoint_convert_dir" \
        --model_name "$asr_model_version" \
        --dtype "$dtype"

    # Step 3: Build TensorRT engine for encoder
    echo "Building TensorRT engine for encoder..."
    trtllm-build \
        --checkpoint_dir "$checkpoint_convert_dir/encoder" \
        --output_dir "$trt_engine_dir/encoder" \
        --moe_plugin disable \
        --max_batch_size "$max_batch_size" \
        --gemm_plugin disable \
        --bert_attention_plugin "$dtype" \
        --max_input_len "$max_input_len" \
        --max_seq_len "$max_seq_len"

    # Step 4: Build TensorRT engine for decoder
    echo "Building TensorRT engine for decoder..."
    trtllm-build \
        --checkpoint_dir "$checkpoint_convert_dir/decoder" \
        --output_dir "$trt_engine_dir/decoder" \
        --moe_plugin disable \
        --max_beam_width 1 \
        --max_batch_size "$max_batch_size" \
        --max_seq_len "$decoder_seq_len" \
        --max_input_len "$decoder_input_len" \
        --max_encoder_input_len "$max_input_len" \
        --gemm_plugin "$dtype" \
        --bert_attention_plugin "$dtype" \
        --gpt_attention_plugin "$dtype"

    target_dir="$trt_engine_dir"
    # Copy configuration files
    echo "Copying configuration files from $hf_model_dir to $target_dir..."
    for ext in json txt; do
        find "$hf_model_dir" -name "*.$ext" -exec cp {} "$target_dir/" \;
    done


else
    echo "Downloading model..."
    python /workspace/hf_download_repo.py \
    --repo_id ${org}/${model_name} \
    --revision ${model_version} \
    --local_dir "${base_dir}/base_models" \
    --token ${HUGGING_FACE_HUB_TOKEN}

    echo "Building LLM model..."
    
    set -x
    # Directory structure for LLM
    if [[ "$quantize_fp8_flag" == "true" ]]; then
        checkpoint_dir="$base_dir/convert_models/${model_name}-${model_version}-FP8-TP${tp_size}"
        trt_output_dir="$base_dir/trt_engines/${model_name}-${model_version}-mxbsz${max_batch_size}-ip${max_input_len}-seq${max_seq_len}/fp8/${tp_size}-gpu/trtllm_engine"
    else
        checkpoint_dir="$base_dir/convert_models/${model_name}-${model_version}-BF16-TP${tp_size}"
        trt_output_dir="$base_dir/trt_engines/${model_name}-${model_version}-mxbsz${max_batch_size}-ip${max_input_len}-seq${max_seq_len}/bf16/${tp_size}-gpu/trtllm_engine"
    fi

    model_dir="${base_dir}/base_models/${model_name}/${model_version}"

    # FP8 quantization path
    if [[ "$quantize_fp8_flag" == "true" ]]; then
        echo "Quantizing model to FP8..."
        python /app/tensorrt_llm/examples/quantization/quantize.py \
            --model_dir "$model_dir" \
            --dtype float16 \
            --qformat fp8 \
            --kv_cache_dtype fp8 \
            --output_dir "$checkpoint_dir" \
            --calib_size 512 \
            --tp_size "$tp_size"

        trtllm-build \
            --checkpoint_dir "$checkpoint_dir" \
            --output_dir "$trt_output_dir" \
            --max_batch_size "$max_batch_size" \
            --max_input_len "$max_input_len" \
            --max_seq_len "$max_seq_len" \
            --context_fmha enable \
            --kv_cache_type=paged \
            --remove_input_padding enable \
            --use_fused_mlp enable \
            --gemm_swiglu_plugin fp8 \
            --use_fp8_context_fmha enable \
            --use_paged_context_fmha enable \
            --gemm_plugin float16 \
            --gpt_attention_plugin float16 \
            --workers 1

    # BF16 conversion path
    else
        echo "Converting model to BF16..."
        python /app/tensorrt_llm/examples/${base_model}/convert_checkpoint.py \
            --model_dir "$model_dir" \
            --output_dir "$checkpoint_dir" \
            --dtype bfloat16 \
            --tp_size "$tp_size"

        trtllm-build \
            --checkpoint_dir "$checkpoint_dir" \
            --output_dir "$trt_output_dir" \
            --max_batch_size "$max_batch_size" \
            --max_input_len "$max_input_len" \
            --max_seq_len "$max_seq_len" \
            --context_fmha enable \
            --use_paged_context_fmha enable \
            --kv_cache_type=paged \
            --remove_input_padding enable \
            --use_fused_mlp enable \
            --gemm_plugin bfloat16 \
            --gpt_attention_plugin bfloat16
    fi

    target_dir=$(dirname "$trt_output_dir")
    # Copy JSON and TXT files
    echo "Copying configuration files from $model_dir to $target_dir..."
    for filename in "$model_dir"/*.json; do
        cp "$filename" "$target_dir/"
    done

    for filename in "$model_dir"/*.txt; do
        cp "$filename" "$target_dir/"
    done
fi

# Upload the model (common for both paths)
echo "Uploading model..."
set +e
azcopy copy --recursive --overwrite true "$target_dir" "$base_storage_path/${model_name}-${model_version}-bs${max_batch_size}"
exit_code=$?
set -e

if [[ $exit_code -ne 0 ]]; then
    echo "Engine built in $target_dir. Azcopy failed with exit code $exit_code. Check if files are already present."
else
    echo "Model successfully processed and uploaded!"
fi
