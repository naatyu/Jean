"""Offline checks using a tiny real Liquid architecture, never downloaded weights."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from PIL import Image
from transformers import (
    Lfm2Config,
    Lfm2VlConfig,
    Lfm2VlForConditionalGeneration,
    Siglip2VisionConfig,
)

from jean.model import Jean, answer_codes
from jean.train import fit


class Processor:
    """Minimal test processor; image patches still pass through the real vision tower."""

    tokenizer = None

    def __init__(self) -> None:
        self.tokenizer = self

    def encode(self, text: str, **kwargs: object) -> list[int]:
        text = text.strip()
        if text in {"AA", "AB"}:
            return [28 if text == "AA" else 29]
        if text.isdigit():
            return [11 + int(text)]
        binary = {"yes": 5, "Yes": 6, "YES": 7, "no": 8, "No": 9, "NO": 10}
        return [binary[text]] if text in binary else [ord(text) - ord("A") + 2]

    def get_vocab(self) -> dict[str, int]:
        return {**{chr(ord("A") + i): i + 2 for i in range(26)}, "AA": 28, "AB": 29}

    def decode(self, ids: list[int]) -> str:
        return next(
            code for code, token_id in self.get_vocab().items() if token_id == ids[0]
        )

    def apply_chat_template(self, messages: list, **kwargs: object) -> str:
        return messages[0]["content"][-1]["text"]

    def __call__(self, text: list[str], images: list | None, **kwargs: object) -> dict:
        result = {"input_ids": torch.tensor([[1, 2, 3]])}
        if images:
            result.update(
                input_ids=torch.tensor([[63, 2, 3]]),
                pixel_values=torch.ones(1, 4, 12),
                pixel_attention_mask=torch.ones(1, 4, dtype=torch.long),
                spatial_shapes=torch.tensor([[2, 2]]),
            )
        return result

    def save_pretrained(self, path: str) -> None:
        """Checkpoint test substitutes this processor when restoring weights."""
        pass


def tiny_model() -> Jean:
    """Use actual convolution, attention and vision layers without large weights."""
    text = Lfm2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        layer_types=["conv", "full_attention"],
        pad_token_id=0,
    )
    vision = Siglip2VisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        patch_size=2,
        num_patches=4,
    )
    config = Lfm2VlConfig(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        image_token_id=63,
        downsample_factor=2,
    )
    return Jean(Lfm2VlForConditionalGeneration(config), Processor())


def test_text_image_training_and_checkpoint(tmp_path: Path) -> None:
    model = tiny_model()
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "red").save(image)
    decision = {
        "type": "choice",
        "state": "Use the image",
        "question": "Which color?",
        "choices": {"red": "Red", "blue": "Blue"},
        "target": "red",
        "images": [str(image)],
    }
    assert model(decision).shape == (2,)
    model(decision).sum().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.model.model.vision_tower.parameters()
    )
    before = model.model.lm_head.weight.detach().clone()
    fit(model, [decision], epochs=1, lr=1e-3)
    assert not torch.equal(before, model.model.lm_head.weight)
    expected = model.predict(decision)
    assert abs(sum(expected["probabilities"].values()) - 1) < 1e-6
    model.save(str(tmp_path / "checkpoint"))
    with patch("jean.model.AutoProcessor.from_pretrained", return_value=Processor()):
        restored = Jean.load(str(tmp_path / "checkpoint"), device="cpu")
    assert restored.predict(decision) == expected
    text_decision = {k: v for k, v in decision.items() if k != "images"}
    assert restored(text_decision).shape == (2,)


def test_mixed_decision_training_and_outputs() -> None:
    model = tiny_model()
    common = {
        "type": "choice",
        "state": "A customer requests a refund.",
        "question": "Decide",
    }
    decisions = [
        {
            **common,
            "choices": {"billing": "Billing", "support": "Support"},
            "target": "billing",
        },
        {
            **common,
            "type": "noul",
            "choices": {"no": "No", "yes": "Yes"},
            "target": "yes",
        },
        {
            **common,
            "type": "score",
            "choices": {"0": "Low", "1": "Medium", "2": "High"},
            "target": "1",
        },
    ]
    fit(model, decisions, epochs=1, lr=1e-3)
    for decision in decisions:
        result = model.predict(decision)
        assert set(result["probabilities"]) == set(decision["choices"])
        assert abs(sum(result["probabilities"].values()) - 1) < 1e-6
        if decision["type"] == "noul":
            assert result["noul"] == result["probabilities"]["yes"]
        elif decision["type"] == "score":
            assert 0 <= result["score"] <= 2
    # Check rating semantics: expected value, rather than the most likely level.
    score_decision = next(
        decision for decision in decisions if decision["type"] == "score"
    )
    with patch.object(
        model, "forward", return_value=torch.tensor([0.1, 0.6, 0.3]).log()
    ):
        assert model.predict(score_decision)["score"] == pytest.approx(1.2)
    with pytest.raises(ValueError):
        answer_codes(
            {**score_decision, "choices": {"0": "Low", "2": "High"}}, model.choice_codes
        )
    with pytest.raises(ValueError):
        answer_codes(
            {**common, "type": "noul", "choices": {"true": "Yes", "false": "No"}},
            model.choice_codes,
        )

    with pytest.raises(KeyError, match="type"):
        answer_codes(
            {k: v for k, v in score_decision.items() if k != "type"}, model.choice_codes
        )


def test_choices_continue_after_z() -> None:
    model = tiny_model()
    decision = {
        "type": "choice",
        "state": "Pick a category",
        "question": "Which?",
        "choices": {str(i): f"Category {i}" for i in range(28)},
    }
    assert answer_codes(decision, model.choice_codes)[-2:] == ["AA", "AB"]
    logits = model(decision)
    assert logits.shape == (28,)
    assert len({ids[0] for ids in model.answer_tokens(model.choice_codes)}) == 28
    logits[-1].backward()
    assert model.model.lm_head.weight.grad is not None
