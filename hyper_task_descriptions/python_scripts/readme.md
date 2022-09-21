# Python Scripts

`p3_results.py` - script for displaying results from evaluation easy.
`fixed_roberta.py` - make and upload the slightly altered roberta model.

## poking_the_bear.py

This is a fun little script for testing pretrained models. First, download a checkpoint locally (this may take a while). Make sure you are in the right env (install requrements and add the repo to your python path):
```
pip install -r requirements.txt
export PYTHONPATH=$(pwd)
```

You can then run the script as follows:
```
python hyper_task_descriptions/python_scripts/poking_the_bear.py -t <checkpoint_dir>
```

This will then put you into a `pdb` shell after loading, and I have provided two functions to show how to play: `get_out` and `get_attn`, which get the hyperencoder output and the cross-attention probabilities respectively. Feel free to mess around and add your own functions etc. I was lazy here so if you want to emulate different model output processing, you'll have to code that up yourself.
