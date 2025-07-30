
# TensorRT-LLM Whisper Setup and Usage Guide


## Setup Instructions

### 1. Clone the Repository

```bash
git clone git@github.com:sarvamai/sarvam-nim.git
```

Replace the ```build_and_upload.sh``` file in sarvam-nim/tools with the corresponding file in this repo
Replace  the ```.env-build-asr``` in sarvam-nim/tools/examples with the corresponding file in this repo

### 2. Docker Commands
```bash
docker pull appsprodacr.azurecr.io/trt-llm-whisper:latest
docker run --rm -it \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --gpus device=0 \
  -v <path to sarvam nim>:/inference \
  -v <path to store trt engine>:/models \
  -e HUGGING_FACE_HUB_TOKEN=<your_hf_token> \
  --env-file <path to sarvam-nim>/tools/examples/.env-build-asr \
  appsprodacr.azurecr.io/trt-llm-whisper:latest
```
### 3. Building the TRT Engines
```bash
cd /inference/tools && bash build_and_upload.sh
```
### 4. Running inference
```bash
cd /app/tensorrt_llm/examples/models/core/whisper
```
**Single Audio Inference**
```bash
python3 run.py --name single_wav_test --engine_dir /models/trt_engines/saaras-raft-wp20-base-v2v-v2-chunk_5-main-bs72/1-gpu --input_file <path-to-audio>.wav
```
**On a hf dataset**
```bash
python3 run.py --engine_dir  /models/trt_engines/saaras-raft-wp20-base-v2v-v2-chunk_5-main-bs72/1-gpu  --dataset hf-internal-testing/librispeech_asr_dummy --enable_warmup --name librispeech_dummy_large_v3
```
