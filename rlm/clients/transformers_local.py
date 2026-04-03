from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from typing import Any

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary

_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


class TransformersClient(BaseLM):
    def __init__(
        self,
        model_name: str,
        model_path: str | None = None,
        *,
        device: str = "cuda:0",
        device_map: str | dict[str, Any] | None = None,
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        attn_implementation: str | None = None,
        enable_thinking: bool = False,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        tokenizer: Any | None = None,
        model: Any | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.model_path = model_path or model_name
        self.device = device
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.attn_implementation = attn_implementation
        self.enable_thinking = enable_thinking
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.tokenizer = tokenizer
        self.model = model
        self.tokenizer_kwargs = dict(tokenizer_kwargs or {})
        self.model_kwargs = dict(model_kwargs or {})

        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

    def completion(self, prompt: str | list[dict[str, Any]] | dict[str, Any]) -> str:
        self._ensure_loaded()
        messages = self._normalize_messages(prompt)
        tokenizer = self.tokenizer
        model = self.model
        if tokenizer is None or model is None:
            raise ValueError("TransformersClient was not initialized correctly")

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        model_inputs = tokenizer([text], return_tensors="pt")
        input_device = self._resolve_input_device()
        if hasattr(model_inputs, "to"):
            model_inputs = model_inputs.to(input_device)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p

        generated_ids = model.generate(**model_inputs, **generation_kwargs)
        input_length = int(model_inputs.input_ids.shape[-1])
        output_ids = generated_ids[0][input_length:]
        content = tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        content = _THINK_PATTERN.sub("", content).strip()

        self.model_call_counts[self.model_name] += 1
        self.model_input_tokens[self.model_name] += input_length
        self.model_output_tokens[self.model_name] += int(output_ids.shape[-1])
        self.last_prompt_tokens = input_length
        self.last_completion_tokens = int(output_ids.shape[-1])
        return content

    async def acompletion(self, prompt: str | list[dict[str, Any]] | dict[str, Any]) -> str:
        return await asyncio.to_thread(self.completion, prompt)

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(
            model_usage_summaries={
                model_name: ModelUsageSummary(
                    total_calls=self.model_call_counts[model_name],
                    total_input_tokens=self.model_input_tokens[model_name],
                    total_output_tokens=self.model_output_tokens[model_name],
                )
                for model_name in self.model_call_counts
            }
        )

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=self.last_prompt_tokens,
            total_output_tokens=self.last_completion_tokens,
        )

    def _ensure_loaded(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            **self.tokenizer_kwargs,
        }
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            **self.model_kwargs,
        }
        if self.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.attn_implementation

        import torch

        model_kwargs["torch_dtype"] = _resolve_torch_dtype(torch, self.torch_dtype)
        model_kwargs["device_map"] = self.device_map or self.device

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **tokenizer_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)

    def _normalize_messages(
        self, prompt: str | list[dict[str, Any]] | dict[str, Any]
    ) -> list[dict[str, Any]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        if isinstance(prompt, dict):
            return [prompt]
        if isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            return prompt
        raise ValueError(f"Invalid prompt type for TransformersClient: {type(prompt)}")

    def _resolve_input_device(self):
        if self.model is None:
            raise ValueError("Model is not loaded")
        try:
            return next(self.model.parameters()).device
        except StopIteration as exc:
            raise ValueError("Transformers model has no parameters") from exc


def _resolve_torch_dtype(torch_module, value: str | Any):
    if not isinstance(value, str):
        return value
    if value == "auto":
        return "auto"
    if not hasattr(torch_module, value):
        raise ValueError(f"Unsupported torch dtype: {value}")
    return getattr(torch_module, value)
