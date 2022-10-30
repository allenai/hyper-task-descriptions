export PYTHONPATH=${PYTHONPATH}:${PWD}
# setup t5x (important)
git clone --branch=main https://github.com/google-research/t5x # TODO: pin to specific commit.
cd t5x
python3 -m pip install -e '.[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# install promptsource, and fix the seqio dependency
python3 -m pip install promptsource
# i use a new feature in t5.data
python3 -m pip uninstall -y t5
python3 -m pip install git+https://github.com/google-research/text-to-text-transfer-transformer.git
# use a compatible version of optax
python3 -m pip uninstall -y optax
python3 -m pip install optax==0.1.2
# custom fixed seqio
python3 -m pip uninstall -y seqio seqio-nightly
python3 -m pip install git+https://github.com/hamishivi/seqio.git

# I've had some issues with tensorflow. these versions seem to work
python3 -m pip install tensorflow==2.9.0
python3 -m pip install tensorflow-text==2.9.0
echo "----- ALL DEPENDENCIES INSTALLED -----"
# next, we cache the tokenizers / HF splits used so we don't have to load them later.
# This can take ~15 minutes.
python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('t5-base')"
python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('google/t5-large-lm-adapt')"
python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('google/t5-small-lm-adapt')"
# TRANSFORMERS_OFFLINE=1 python3 -c "import hyper_task_descriptions.seqio_tasks.all_t0_tasks"
echo "----- CACHED TOKENIZERS AND SPLITS -----"
# and we are done!
echo "----- TPU SETUP COMPLETE -----"
