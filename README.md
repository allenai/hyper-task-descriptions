# Hyper-Task-Descriptions

Repo for learning adapters + prefixes from task descriptions :)

## Installation

Before running any code, you'll need access to `gs://hamishi-tpu-bucket` (where I have put cached P3 data). If you have an allenai gcloud account this should be enough, just make sure to login with `gcloud auth application-default login` so t5x uses your credentials (I think).
### Local Installation

`pip install -e .` should work. If not, the main two dependencies are [`t5x`](https://github.com/google-research/t5x) and [`promptsource`](https://github.com/bigscience-workshop/promptsource), which you can find installation instructions for at the links. Note we don't need to run the interface bundled with promptsource, so you can safely ignore the python 3.7 requirement.

### TPU installation

Please install t5x using the instructions found [here](https://github.com/google-research/t5x#installation) - there are some TPU-specific things to install with T5X. Afterwards, install promptsource.

## Running

Run `./scripts/local.sh` for a small model + small subset of T0 data that is useful for local development.

`script/t0_train.sh` will train a T0 model.

`script/t0_eval.sh` will run evaluation (note: I have not tested this yet).

These scripts are fairly simple, so please look at them to determine how to run more custom samples. Note that despite caching P3 data, the code will still need to make internet connections to get dataset sizes/splits/etc. You only have to let this run through once, and afterwards you can use `HF_DATASETS_OFFLINE=1` to force the model to use its cache, which vastly speeds things up.
