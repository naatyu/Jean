"""Run Jean through the official Decision Index engine interface."""

from typing import Any

import torch
from decision_index.engines import Engine, Unsupported

from .model import BASE_MODEL, Decision, Jean


class JeanEngine(Engine):
    """Translate benchmark questions; leave running and scoring to Decision Index."""

    name = "jean"

    def __init__(
        self, model: str = BASE_MODEL, device: str = "auto", **options: Any
    ) -> None:
        super().__init__(**options)
        self.jean = Jean.load(model, device=device)
        self.device = next(self.jean.parameters()).device
        self.provenance = {"model": model, "device": str(self.device)}

    def __call__(
        self, state: Any, questions: dict[str, dict[str, Any]]
    ) -> tuple[dict[str, Any], None]:
        """Answer each question without changing its instructions or options."""
        answers = {}
        for key, question in questions.items():
            kind = question["type"]
            if kind == "choice":
                choices = question["criteria"]
                if not 2 <= len(choices) <= len(self.jean.choice_codes):
                    raise Unsupported("Unsupported number of choices")
            elif kind == "noul":
                choices = {"yes": "Yes", "no": "No"}
            else:
                raise Unsupported(f"Unsupported question type: {kind}")
            decision: Decision = {
                "type": kind,
                "state": state,
                "question": question["instructions"],
                "choices": choices,
            }
            try:
                answers[key] = self.jean.predict(decision)
            except ValueError as error:
                # Capacity refusals are final; actual failures remain retryable errors.
                if str(error).startswith("Input exceeds 4096 tokens"):
                    raise Unsupported(str(error)) from error
                raise
        return {"model": self.name, "answers": answers}, None

    def synchronize(self) -> None:
        """Include completed GPU work in the runner's latency measurements."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()
