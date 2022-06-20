"""
I want to use a roberta encoder, but t5x needs internal parameters to be divisible by 2/4/8.
This messes with the token type and word embeddings, which are not even.
This script just fixes these things and then uploads to huggingface.
"""
from transformers import FlaxRobertaModel, RobertaModel

model = RobertaModel.from_pretrained(
    "roberta-large", type_vocab_size=8, max_position_embeddings=520, ignore_mismatched_sizes=True
)

# resize vocab to be a little smaller
if model.config.vocab_size % 8 > 0:
    model.resize_token_embeddings(model.config.vocab_size + (8 - model.config.vocab_size % 8))

# token_type_embeddings has vocab size 1 so it wont be inited above. instead copy a pretrained matrix over
pt_model = RobertaModel.from_pretrained("roberta-large")
model.embeddings.token_type_embeddings.weight.data.copy_(
    pt_model.embeddings.token_type_embeddings.weight.data
)
for i in range(0, 8):
    assert (
        model.embeddings.token_type_embeddings.weight.data[0]
        == pt_model.embeddings.token_type_embeddings.weight.data
    ).all()
model.embeddings.position_embeddings.weight.data[
    :514, :
] = pt_model.embeddings.position_embeddings.weight.data
assert (
    model.embeddings.position_embeddings.weight.data[:514]
    == pt_model.embeddings.position_embeddings.weight.data
).all()

# save our model in flax
model.save_pretrained("fixed-roberta-large")
flax_model = FlaxRobertaModel.from_pretrained("fixed-roberta-large", from_pt=True)
flax_model.push_to_hub(
    "hamishivi/fixed-roberta-large",
    token="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",
    use_temp_dir=True,
)
