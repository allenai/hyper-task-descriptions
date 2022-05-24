export PYTHONPATH=${PYTHONPATH}:${PWD}
# setup t5x (important)
git clone --branch=main https://github.com/google-research/t5x # TODO: pin to specific commit.
cd t5x
python3 -m pip install -e '.[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# install promptsource, and fix the seqio dependency
# we install a custom fixed seqio.
python3 -m pip install promptsource
python3 -m pip uninstall -y seqio seqio-nightly
python3 -m pip install git+https://github.com/hamishivi/seqio.git
# I've had some issues with tensorflow. these versions seem to work
python3 -m pip install tensorflow==2.9.0
python3 -m pip install tensorflow-text==2.9.0
echo "----- ALL DEPENDENCIES INSTALLED -----"
# next, we cache the tokenizers / HF splits used so we don't have to load them later.
# This can take ~15 minutes.
python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('t5-base')"
python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('roberta-base')"
python3 -c "from transformers import FlaxRobertaModel; FlaxRobertaModel.from_pretrained('hamishivi/fixed-roberta-base')"
TRANSFORMERS_OFFLINE=1 python3 -c "import hyper_task_descriptions.seqio_tasks.all_t0_tasks"
echo "----- CACHED TOKENIZERS AND SPLITS -----"
# and we are done!
echo "----- TPU SETUP COMPLETE -----"
