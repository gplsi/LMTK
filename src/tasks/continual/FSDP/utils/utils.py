from transformers.models.llama.modeling_llama import LlamaDecoderLayer


# TODO: Add more wrappers for other models, and make clear the keys for the wrappers

AUTO_WRAPPER = {
    "llama": LlamaDecoderLayer,
}