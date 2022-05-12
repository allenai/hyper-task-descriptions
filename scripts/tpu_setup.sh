# setup t5x (important)
git clone --branch=main https://github.com/google-research/t5x # TODO: pin to specific commit.
cd t5x
python3 -m pip install -e '.[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# install promptsource, and fix the seqio dependency (we want seqio-nightly)
python3 -m pip install promptsource
python3 -m pip uninstall -y seqio seqio-nightly
python3 -m pip install seqio-nightly # TODO: pin to a specific version
# and we are done!
echo "TPU setup finished."