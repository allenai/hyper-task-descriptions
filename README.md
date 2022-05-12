# Hyper-Task-Descriptions

Repo for learning adapters + prefixes from task descriptions :)

## Installation

Before running any code, you'll need access to `gs://hamishi-tpu-bucket` (where I have put cached P3 data). If you have an allenai gcloud account this should be enough, just make sure to login with `gcloud auth application-default login` so t5x uses your credentials (I think).
### Local Installation

`pip install -e .` should work. If not, the main two dependencies are [`t5x`](https://github.com/google-research/t5x) and [`promptsource`](https://github.com/bigscience-workshop/promptsource), which you can find installation instructions for at the links. Note we don't need to run the interface bundled with promptsource, so you can safely ignore the python 3.7 requirement.

**N.B. If you want to run locally without cached tasks (e.g. you might be changing the preprocessing) then you'll need to make the changes made in [this seqio PR](https://github.com/google/seqio/pull/153) in your copy of seqio.**

### TPU installation

See below for instructions on setting up and running on TPUs.

## Running

Run `./scripts/local.sh` for a small model + small subset of T0 data that is useful for local development.

`script/t0_train.sh` will train a T0 model.

`script/t0_eval.sh` will run evaluation.

These scripts are fairly simple, so please look at them to determine how to run more custom samples. Note that despite caching P3 data, the code will still need to make internet connections to get dataset sizes/splits/etc. You only have to let this run through once, and afterwards you can use `HF_DATASETS_OFFLINE=1` to force the model to use its cache, which vastly speeds things up.

Note that tensorboard is integrated into t5x, so you should be able to launch tensorboard locally and point it at the bucket directory where you are outputting model artifacts to monitor training/evaluation.

### Running on the TPU

Running on TPU slices is a bit of a pain as you cannot 'just ssh to the machine and run stuff directly'. Rather, you send commands to all the TPUs, and Jax/Flax/t5x works it all out for you in the background. Here's a rough guide of useful steps for setting up and running directly on a TPU.

First, create your tpu! I'll leave aside some of the details but you can do this with a command like:
```
gcloud alpha compute tpus tpu-vm create <name> --accelerator-type=<tpu-version> --zone=<zone> --project=<project> --version=<software-version>
```

You can get some details on TPU architectures [here](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm) and software versions [here](https://cloud.google.com/tpu/docs/supported-tpu-versions). Note that the software version shouldn't matter that much for us, since we're using `jax`.

Next, we need to setup our tpu. Note that if you're using a single TPU (i.e. a `v3-8`), you can ssh directly to the TPU vm with `gcloud alpha compute tpus tpu-vm ssh <tpu-name> --zone=<zone> --project=<project>` and do all your setup directly on the box. However, anything larger requires managing multiple machines and sending shell commands using a command like `gcloud alpha compute tpus tpu-vm ssh <tpu-name> --zone=<zone> --project=<project> --worker=all --command="<bash commands>"`. If even one machine fails when you run the command, try again - you need them all basically to be setup the same way. The rest of this guide will assume you're using a TPU slice, but should also work on a single `v3-8` (although directly ssh'ing to the box is probably easier if you're testing stuff out).

To setup our TPUs, just clone this repo and run `scripts/tpu_setup.sh` as such: 
```bash
gcloud alpha compute tpus tpu-vm ssh <tpu-name> --zone=<zone> --project=<project> --worker=all --command="git clone https://github.com/allenai/hyper-task-descriptions.git; ./hyper-task-descriptions/scripts/tpu_setup.sh"
```
Refer to comments in that script in case something fails (although fingers crossed nothing does!). Note that this is currently a private repo so you'll probably have to use a github authentication token and alter the url accordingly.

Then we can run our model with the following:
```bash
gcloud alpha compute tpus tpu-vm ssh <tpu-name> --zone=<zone> --project=<project> --worker=all --command="cd hyper-task-descriptions; ./scripts/<script-name>
```
This will run the given script on all TPUs. Note this will run the script on all TPUs at once, so you will see a lot of output being logged. If one TPU errors the rest will continue to run, so cancel the command (control+C) and **before rerunning follow the cleanup steps below**.

### TPU Troubleshooting

#### Rerunning

When rerunning code on the TPUs, you need to make sure there are no processes using a core on *any* TPU, otherwise things wont work. To check, run:
```bash
gcloud alpha compute tpus tpu-vm ssh <tpu-name> --zone=<zone> --project=<project> --worker=all --command="sudo lsof -w /dev/accel0"
```
This will give you some PIDs you can then kill with e.g. ```gcloud alpha compute tpus tpu-vm ssh <tpu-name> --zone=<zone> --project=<project> --worker=all --command="kill -9 <PID>"``` (there is probably a better way to do this, but this worked for me!)

Sometimes there can be random lockfiles that hold TPUs too if you aborted a Jax program early (see [here](https://github.com/google/jax/issues/10192)). Remove them with 
```bash
gcloud alpha compute tpus tpu-vm ssh <tpu-name> --zone=<zone> --project=<project> --worker=all --command="sudo rm -f /tmp/libtpu_lockfile; sudo rm -rf /tmp/tpu_logs"
```
#### Altering scripts / code

Sometimes you might want to run or hotfix some script or file in this repo after scaling up to multiple TPUs. To do so, after everything is setup, create the new/edited file locally. Then copy it over to the TPUs with:
```bash
gcloud alpha compute tpus tpu-vm scp <new-file>  <tpu-name>: --zone=<zone> --project=<project> --worker=all
```
You might be able to specify the destination path after <tpu-name> but it didn't work when I tried it. Instead, I just use a `mv` command to move the file where it should go.

**Note: you might get weird `ssh` errors when running the above `scp` command.** If you do, run `ssh-add .../.ssh/google_compute_engine` like the error probably suggests and rerun the command right after. Sometimes this takes a few tries before it runs without error, although so long as the new file ends up on all TPUs you're good to go.

After moving the file to where it should go, you're done! Run the script or rerun your model or whatever you need to do.
