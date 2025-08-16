# Start from the NVIDIA PyTorch image
FROM nvcr.io/nvidia/tensorrt-llm/release:0.20.0rc4
WORKDIR /workspace
# Set environment variables
ENV PYTHONUNBUFFERED="1"
ENV PYTHONIOENCODING="utf-8"
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Upgrade pip
RUN python3 -m pip install --upgrade pip
RUN pip install  evaluate~=0.4.1
RUN pip install rouge_score~=0.1.2
RUN pip install kaldialign
RUN pip install openai-whisper

# Expose the ports
EXPOSE 8000 8001 8002

# Command to run when the container starts
CMD ["/bin/bash"]
