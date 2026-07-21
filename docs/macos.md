# Running Loghi-HTR natively on macOS (Apple Silicon)

Loghi-HTR inference runs natively on Apple Silicon via the
[tensorflow-metal](https://developer.apple.com/metal/tensorflow-plugin/)
plugin, which registers the Apple GPU as a regular TensorFlow GPU device.
No Docker or CUDA required.

## Requirements

- Apple Silicon Mac (M1 or newer)
- Python 3.12+
- Xcode command line tools (`xcode-select --install`) — needed to build the
  word-beam-search extension from source

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-macos.txt
```

## Run the API

The existing `--gpu` handling works unchanged: tensorflow-metal exposes the
Apple GPU at index 0, so the default `--gpu 0` selects it and `--gpu -1`
forces CPU.

```bash
export LOGHI_MODEL_PATH=/path/to/model
export LOGHI_OUTPUT_PATH=/path/to/output
export LOGHI_BATCH_SIZE=32
export LOGHI_GPUS=0
./src/api/start_local_app.sh
```

## Verifying correctness

tensorflow-metal has a history of op-level bugs that produce silently wrong
results. Before trusting GPU output, transcribe a few pages with `--gpu 0`
and again with `--gpu -1` (CPU) and diff the text. If they disagree, run on
CPU or disable mixed precision with `--use_float32`.

## Notes

- Mixed precision (`mixed_float16`) is enabled by default when a GPU is
  detected; pass `--use_float32` to disable it if you see degraded output.
- Training on Metal is untested; this port targets inference.
