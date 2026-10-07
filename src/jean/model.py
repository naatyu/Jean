"""One-pass decisions using Liquid's existing vocabulary head."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, NotRequired, Self, TypedDict

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

BASE_MODEL = "LiquidAI/LFM2.5-VL-3B"


class Decision(TypedDict):
    """The same small record format for every dataset and prediction."""

    type: Literal["choice", "noul", "score"]
    state: Any
    question: str
    choices: dict[str, str]
    images: NotRequired[list[str]]
    target: NotRequired[str]


def answer_codes(decision: Decision, choice_codes: list[str]) -> list[str]:
    """Match each choice to a letter, yes/no, or an ordered rating digit."""
    kind, choices = decision["type"], decision["choices"]
    if kind == "choice" and 2 <= len(choices) <= len(choice_codes):
        return choice_codes[: len(choices)]
    if kind == "noul" and set(choices) == {"yes", "no"}:
        return list(choices)
    if (
        kind == "score"
        and 2 <= len(choices) <= 10
        and list(choices) == [str(i) for i in range(len(choices))]
    ):
        return list(choices)
    raise ValueError(
        f"Expected choice: 2..{len(choice_codes)} options; noul: yes/no; score: ordered levels 0..N, 2..10 levels"
    )


class Jean(torch.nn.Module):
    """Keep the pretrained model; normalize its answer logits over supplied choices."""

    def __init__(self, model: torch.nn.Module, processor: Any) -> None:
        super().__init__()
        self.model, self.processor = model, processor
        tokenizer = processor.tokenizer
        # Read the vocabulary once; skip letter combinations that split into tokens.
        decoded = {
            tokenizer.decode([i]).removeprefix(" ")
            for i in tokenizer.get_vocab().values()
        }
        candidates = sorted(
            (
                c
                for c in decoded
                if c.isascii() and c.isalpha() and c.isupper() and len(c) <= 3
            ),
            key=lambda c: (len(c), c),
        )
        self.choice_codes = [
            c
            for c in candidates
            if any(
                len(tokenizer.encode(form, add_special_tokens=False)) == 1
                for form in (c, " " + c)
            )
        ]

    @classmethod
    def load(cls, path: str = BASE_MODEL, device: str = "auto") -> Self:
        """Load base weights or our checkpoint."""
        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError(
                "Metal GPU unavailable in this process; use cpu or run outside the sandbox"
            )
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable; run on a GPU machine or choose cpu")
        if path == BASE_MODEL and Path(".cache/liquid/config.json").exists():
            path = ".cache/liquid"  # Reuse the downloaded checkpoint without a second copy.
        model = AutoModelForImageTextToText.from_pretrained(
            path, dtype=torch.float32, attn_implementation="sdpa"
        )
        return cls(model, AutoProcessor.from_pretrained(path)).to(device)

    def forward(self, decision: Decision) -> torch.Tensor:
        """Prepare inputs, run the model, and read the allowed answer scores."""
        codes = answer_codes(decision, self.choice_codes)
        inputs = self.prepare_input(decision, codes)
        token_groups = self.answer_tokens(codes)
        device = next(self.parameters()).device
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = (
                self.model(**inputs, logits_to_keep=1, use_cache=False)
                .logits[0, -1]
                .float()
            )
        return torch.stack([logits[ids].max() for ids in token_groups])

    def prepare_input(
        self, decision: Decision, codes: list[str]
    ) -> dict[str, torch.Tensor]:
        """Render the prompt, process text/images, and move tensors to the device."""
        choices = decision["choices"]
        kind = decision["type"]
        state = (
            decision["state"]
            if isinstance(decision["state"], str)
            else json.dumps(decision["state"], ensure_ascii=False)
        )
        options = "\n".join(
            f"{code}: {text}" for code, text in zip(codes, choices.values())
        )
        reply = (
            "yes or no"
            if kind == "noul"
            else f"a digit 0-{len(choices) - 1}"
            if kind == "score"
            else "the option code"
        )
        prompt = f"State:\n{state}\n\nQuestion:\n{decision['question']}\n\nOptions:\n{options}\n\nReply with {reply} only."
        content = [{"type": "image"} for _ in decision.get("images", [])]
        content.append({"type": "text", "text": prompt})
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        images = []
        for path in decision.get("images", []):
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((1024, 1024))
                images.append(image)
        inputs = self.processor(
            text=[text], images=[images] if images else None, return_tensors="pt"
        )
        if inputs["input_ids"].shape[1] > 4096:
            raise ValueError("Input exceeds 4096 tokens; shorten it explicitly")
        if "pixel_attention_mask" in inputs:
            # Remove padded image patches to keep local image checks affordable.
            count = int(inputs["pixel_attention_mask"].sum(1).max())
            inputs["pixel_values"] = inputs["pixel_values"][:, :count]
            inputs["pixel_attention_mask"] = inputs["pixel_attention_mask"][:, :count]
        device = next(self.parameters()).device
        return {key: value.to(device) for key, value in inputs.items()}

    def answer_tokens(self, codes: list[str]) -> list[list[int]]:
        """Find each answer's single-token forms, pooling spelling variants."""
        # Pool single-token spelling variants, then restrict softmax to these answers.
        groups = []
        for code in codes:
            variants = [
                self.processor.tokenizer.encode(form, add_special_tokens=False)
                for form in (
                    (code, f" {code}", code.capitalize(), code.upper())
                    if code in {"yes", "no"}
                    else (code, f" {code}")
                )
            ]
            ids = sorted({tokens[0] for tokens in variants if len(tokens) == 1})
            if not ids:
                raise ValueError(f"No single-token answer code for {code}")
            groups.append(ids)
        return groups

    @torch.inference_mode()
    def predict(self, decision: Decision) -> dict[str, Any]:
        """Return a choice, P(yes), or expected rating, without generating text."""
        self.eval()
        probabilities = dict(
            zip(decision["choices"], self(decision).softmax(-1).tolist())
        )
        kind = decision["type"]
        result = {"type": kind, "probabilities": probabilities}
        if kind == "noul":
            result["noul"] = probabilities["yes"]
        elif kind == "score":
            result["score"] = sum(int(level) * p for level, p in probabilities.items())
        else:
            result["choice"] = max(probabilities, key=probabilities.get)
        return result

    def save(self, path: str) -> None:
        """Save standard model and processor files for Jean.load."""
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
        license_file = Path(__file__).parents[2] / "licenses/LIQUID_LICENSE"
        if license_file.exists():
            (Path(path) / "LICENSE").write_bytes(license_file.read_bytes())
