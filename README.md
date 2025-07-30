
# TensorRT-LLM Whisper Setup and Usage Guide


## Setup Instructions



### 1. Docker Commands
```bash
docker pull appsprodacr.azurecr.io/trt-llm-whisper:latest
docker run --rm -it \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --gpus device=0 \
  -v <path to sarvam nim>:/inference \
  -v <path to  trt engine>:/models \
  -e HUGGING_FACE_HUB_TOKEN=<your_hf_token> \
  --env-file <path to sarvam-nim>/tools/examples/.env-build-asr \
  appsprodacr.azurecr.io/trt-llm-whisper:latest
```

### 2. Running inference
```bash
cd /app/tensorrt_llm/examples/models/core/whisper
```
**Single Audio Inference**
```bash
python3 run.py --name single_wav_test --engine_dir /models/trt_engines/saaras-raft-wp20-base-v2v-v2-chunk_5-main-bs72/1-gpu --input_file <path-to-audio>.wav --results_dir <path>
```
**On a hf dataset**
```bash
python3 run.py --engine_dir  /models/trt_engines/saaras-raft-wp20-base-v2v-v2-chunk_5-main-bs72/1-gpu  --dataset hf-internal-testing/librispeech_asr_dummy --enable_warmup --name librispeech_dummy_large_v3 --results_dir <path>
```
