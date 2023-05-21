export PYTHONPATH=${PYTHONPATH}:${PWD}
# setup t5x (important)
git clone https://github.com/google-research/t5x.git
cd t5x
git checkout 3282da46b4a7e46bc17b96cdb6673a4dd812a1b6
# no deps as t5x used un-pinned library versions from github
# we have the pinned versions in requirements.txt
python3 -m pip install -e '.[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html --no-deps
cd ..
# install deps
python3 -m pip install -r requirements.txt --upgrade
# install tpu-specific jax
python3 -m pip install "jax[tpu]==0.3.23" --upgrade -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
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
