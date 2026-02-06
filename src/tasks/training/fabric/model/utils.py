from transformers import PreTrainedModel


# Fabric models always wrap a HF PreTrainedModel (regardless of the specific Auto*
# factory used to build it). Checking against PreTrainedModel covers all concrete
# causal/MLM classes returned by the auto factories while still keeping the guard
# strict enough to catch completely unsupported model types.
AVAILABLE_MODELS = (
    PreTrainedModel,
)
