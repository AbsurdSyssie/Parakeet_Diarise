"""NeMo ASR model loading."""

from __future__ import annotations

import nemo.collections.asr as nemo_asr

from .types import ASRConfig


def load_nemo_asr_model(config: ASRConfig):
    kwargs = {"model_name": config.model_name}
    if config.hf_token:
        kwargs["use_auth_token"] = config.hf_token

    try:
        return nemo_asr.models.ASRModel.from_pretrained(**kwargs)
    except TypeError:
        kwargs.pop("use_auth_token", None)
        return nemo_asr.models.ASRModel.from_pretrained(**kwargs)
