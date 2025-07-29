1. Docker Image: appsprodacr.azurecr.io/trt_llm_whisper:latest
2. Docker Command: docker run --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus device=6 -v /shared/home/sapna/sarvam-nim:/inference -e -v /shared/home/sapna/whisper:/models  HUGGING_FACE_HUB_TOKEN=<your hf token> --env-file sarvam-nim/tools/examples/.env-build-asr  nvcr.io/nvidia/tensorrt-llm/release:1.0.0rc4
3. cd /inference/tools && bash build_and_upload.sh
4. cd /app/tensorrt_llm/examples/models/core/whisper
5. python3 run.py --name single_wav_test --engine_dir /inference/1-gpu --input_file <path-to-audio>.wav
6. python3 run.py --engine_dir $output_dir --dataset hf-internal-testing/librispeech_asr_dummy --enable_warmup --name librispeech_dummy_large_v3



