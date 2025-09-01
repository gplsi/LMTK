from typing import Tuple, Dict, Any
import lightning as L
from transformers import AutoModelForMaskedLM, RobertaConfig, RobertaForMaskedLM, AutoConfig
import torch
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW
from utils.logging import VerboseLevel, get_logger
from src.tasks.training.fabric.model.base import BaseModel
import logging

"""
The LightningModules used for masked language model training.
This module defines MLM models that extend LightningModule for training,
validation, and testing using a pre-trained masked language model with pre-tokenized
datasets that include proper MLM masking.
"""


# Base class for Masked Language Model training with Fabric
class FabricMLM(BaseModel):
    """
    PyTorch Lightning Module for Masked Language Model Training with Fabric.

    This class wraps a Hugging Face pre-trained masked language model and integrates
    it into a LightningModule for MLM training. It handles pre-tokenized datasets
    where tokens are randomly masked for the model to predict. Unlike causal LM,
    MLM can see the entire sequence context when predicting masked tokens.
    It leverages Fabric's distributed environment attributes for enhanced logging during training.
    """    

    def __init__(self, **kwargs) -> None:
        """
        Initialize the FabricMLM model.

        Loads a pre-trained AutoModelForMaskedLM model using the specified parameters.
        Determines the torch data type based on the 'precision' key in kwargs,
        configures logging, and stores additional arguments for future use.
        Optimized for masked language model training with pre-tokenized datasets.

        Args:
            **kwargs: Arbitrary keyword arguments that include:
                - model_name (str): Identifier for the pre-trained model.
                - precision (str): Model precision setting ('bf16-true' selects bfloat16, otherwise uses float32).
                - verbose_level (Optional): Logging verbosity level.
                - zero_stage (Optional): DeepSpeed ZeRO stage level.
                - mlm_probability (Optional): Probability of masking tokens (default: 0.15).
                - ignore_index (Optional): Token ID to ignore in loss calculation (default: -100).
        """
        super().__init__(**kwargs)        
        self.mlm_probability = kwargs.get("mlm_probability", 0.15)
        self.ignore_index = kwargs.get("ignore_index", -100)       
        # Optional integration with custom RoBERTa variants and modifications
        roberta_cfg = kwargs.get("roberta", {}) or {}
        modifications_cfg = kwargs.get("modifications", {}) or {}

        use_custom_roberta = bool(getattr(roberta_cfg, "custom", roberta_cfg.get("custom", False)))
        from_scratch = bool(getattr(roberta_cfg, "from_scratch", roberta_cfg.get("from_scratch", True)))

        if use_custom_roberta and from_scratch:
            # Build Roberta config from provided fields; fall back to HF defaults where missing
            def _get(cfg, key, default):
                return getattr(cfg, key, cfg.get(key, default)) if isinstance(cfg, dict) or hasattr(cfg, key) else default
            def _to_int(v, default=None):
                if v is None:
                    return default
                try:
                    return int(v)
                except Exception:
                    try:
                        return int(float(v))
                    except Exception:
                        return default if default is not None else v
            def _to_float(v, default=None):
                if v is None:
                    return default
                try:
                    return float(v)
                except Exception:
                    return default if default is not None else v

            r_config = RobertaConfig(
                vocab_size=_to_int(_get(roberta_cfg, "vocab_size", 50265), 50265),
                hidden_size=_to_int(_get(roberta_cfg, "hidden_size", 768), 768),
                num_hidden_layers=_to_int(_get(roberta_cfg, "num_hidden_layers", 12), 12),
                num_attention_heads=_to_int(_get(roberta_cfg, "num_attention_heads", 12), 12),
                intermediate_size=_to_int(_get(roberta_cfg, "intermediate_size", 3072), 3072),
                hidden_act="gelu",
                hidden_dropout_prob=_to_float(_get(roberta_cfg, "hidden_dropout_prob", 0.1), 0.1),
                attention_probs_dropout_prob=_to_float(_get(roberta_cfg, "attention_dropout_prob", 0.1), 0.1),
                max_position_embeddings=_to_int(_get(roberta_cfg, "max_position_embeddings", 514), 514),
                type_vocab_size=1,
                initializer_range=_to_float(_get(roberta_cfg, "initializer_range", 0.02), 0.02),
                layer_norm_eps=_to_float(_get(roberta_cfg, "layer_norm_eps", 1e-5), 1e-5),
                pad_token_id=1,
                bos_token_id=0,
                eos_token_id=2,
            )
            self.model = RobertaForMaskedLM(r_config)
        else:
            # Honor from_scratch flag even without custom roberta
            if from_scratch:
                cfg = AutoConfig.from_pretrained(self.model_name)
                self.model = AutoModelForMaskedLM.from_config(cfg)
            else:
                # Default: load pretrained weights
                self.model = AutoModelForMaskedLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch_dtype,
                )

        # Apply optional architecture modifications (attention/FFN)
        from src.models.attention.chebyshev_kan import replace_roberta_attention_with_chebyshev  # type: ignore
        from src.models.attention.chebyshev_kan_orthogonal_projection import replace_roberta_all_layers_with_basis  # type: ignore
        from src.models.attention.chebyshev_kan_orthogonal_projection_sparse import replace_roberta_all_layers_with_basis_sparse  # type: ignore
        from src.models.attention.chebyshev_kan_sparse import replace_roberta_attention_with_chebyshev_sparse  # type: ignore
        from src.models.ffn.chebyshev_kan_ffn import replace_roberta_ffn_with_kan  # type: ignore

        if modifications_cfg:
            # Normalize dict-like access
            def mget(container, key, default=None):
                return getattr(container, key, container.get(key, default)) if isinstance(container, dict) or hasattr(container, key) else default

            attn_cfg = mget(modifications_cfg, "replace_attention", {}) or {}
            attn_mode = mget(attn_cfg, "mode", "none")
            # Normalize/derive attention cheb_order if relevant
            try:
                cheb_order = int(mget(attn_cfg, "cheb_order", 5))
            except Exception:
                cheb_order = 5
            # KAN FFN replacement: strictly require replace_ffn.cheb_order
            ffn_cfg = mget(modifications_cfg, "replace_ffn", None)

            if attn_mode == "chebyshev":
                self.model = replace_roberta_attention_with_chebyshev(self.model, order=cheb_order)
                self.cli_logger.info(f"Applied Chebyshev attention (order={cheb_order})")
            elif attn_mode == "orthogonal":
                basis_name = mget(attn_cfg, "basis_name", "dct")
                basis_rank = int(mget(attn_cfg, "basis_rank", 512))
                self.model = replace_roberta_all_layers_with_basis(self.model, max_order=cheb_order, basis_name=basis_name, basis_rank=basis_rank)
                self.cli_logger.info(f"Applied Orthogonal basis-tied attention (order={cheb_order}, basis={basis_name}, rank={basis_rank})")
            elif attn_mode == "orthogonal_sparse":
                basis_name = mget(attn_cfg, "basis_name", "dct")
                basis_rank = int(mget(attn_cfg, "basis_rank", 512))
                sparsity = mget(attn_cfg, "sparsity", "topk")
                topk = int(mget(attn_cfg, "topk", 64))
                window = int(mget(attn_cfg, "window", 128))
                self.model = replace_roberta_all_layers_with_basis_sparse(self.model, max_order=cheb_order, basis_name=basis_name, basis_rank=basis_rank, sparsity=sparsity, topk=topk, window=window)
                self.cli_logger.info(f"Applied Orthogonal sparse attention (order={cheb_order}, basis={basis_name}, rank={basis_rank}, sparsity={sparsity})")
            elif attn_mode == "sparse":
                sparsity = mget(attn_cfg, "sparsity", "topk")
                topk = int(mget(attn_cfg, "topk", 64))
                window = int(mget(attn_cfg, "window", 128))
                self.model = replace_roberta_attention_with_chebyshev_sparse(self.model, order=cheb_order, sparsity=sparsity, topk=topk, window=window)
                self.cli_logger.info(f"Applied Sparse Chebyshev attention (order={cheb_order}, sparsity={sparsity})")

            # Enforce FFN KAN replacement when configured (no fallback)
            if isinstance(ffn_cfg, dict):
                ffn_order_raw = ffn_cfg.get("cheb_order")
                if ffn_order_raw is None:
                    raise ValueError("replace_ffn.cheb_order must be specified for KAN FFN replacement")
                ffn_order = int(ffn_order_raw)
                self.model = replace_roberta_ffn_with_kan(self.model, order=ffn_order)
                self.cli_logger.info(f"Applied KAN FFN replacement (order={ffn_order})")
        
        self.cli_logger.info(f"Initialized MLM model: {self.model_name}")
        self.cli_logger.info(f"MLM probability: {self.mlm_probability}")
        self.cli_logger.info(f"Ignore index for loss calculation: {self.ignore_index}")
        self.cli_logger.info(f"Model dtype: {self.torch_dtype}")
       
    def _log_mlm_metrics(self, batch, loss, step_type="step"):
        masked_tokens = (batch['labels'] != self.ignore_index).sum().item()
        total_tokens = batch['labels'].numel()
        masking_ratio = masked_tokens / total_tokens if total_tokens > 0 else 0
        self.cli_logger.debug(
            f"MLM {step_type}: loss = {loss.item():.4f}, "
            f"masked_tokens = {masked_tokens}, masking_ratio = {masking_ratio:.3f}"
        )

    def training_step(self, batch, *args):
        # Use base class validation and forward pass
        out = super().training_step(batch, *args)
        # MLM-specific logging (debug level)
        if self.cli_logger.getEffectiveLevel() == logging.DEBUG:
            self._log_mlm_metrics(batch, out["loss"], step_type="train")
        return out

    def validation_step(self, batch, *args):
        out = super().validation_step(batch, *args)
        # MLM-specific logging (debug level)
        if self.cli_logger.getEffectiveLevel() == logging.DEBUG:
            self._log_mlm_metrics(batch, out["loss"], step_type="validation")
        return out

    def test_step(self, batch, *args):
        out = super().test_step(batch, *args)
        # MLM-specific logging (debug level)
        if self.cli_logger.getEffectiveLevel() == logging.DEBUG:
            self._log_mlm_metrics(batch, out["loss"], step_type="test")
        return out