FROM node:20-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates python3 python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code \
    && git config --system --add safe.directory '*' \
    && useradd --create-home --shell /bin/bash agent \
    && mkdir -p /home/agent/.claude \
    && chown -R agent:agent /home/agent/.claude \
    # Deliberately no torch/diffusers/transformers/accelerate/compel: tests
    # that need the real ML stack importable are marked `ml` and auto-skip
    # when torch is absent (tests/conftest.py), so the image stays slim and
    # free of ML-package version pins. The full suite runs on the host.
    && pip install --break-system-packages --no-cache-dir \
       pytest mistralai Pillow

USER agent
WORKDIR /workspace
ENTRYPOINT ["sleep", "infinity"]
