"""Small supervised training loop; one example and one question at a time."""

import argparse
import json
import random
from pathlib import Path

import torch

from .model import Decision, Jean


def read_data(path: str) -> list[Decision]:
    """Read JSONL and resolve optional image paths beside the dataset."""
    source = Path(path).resolve()
    decisions = [
        json.loads(line) for line in source.read_text().splitlines() if line.strip()
    ]
    if not decisions:
        raise ValueError("Empty dataset")
    for decision in decisions:
        if decision["type"] not in {"choice", "noul", "score"}:
            raise ValueError("Unknown decision type")
        if decision["target"] not in decision["choices"]:
            raise ValueError("Target must be a choice ID")
        decision["images"] = [
            str((source.parent / p).resolve()) for p in decision.get("images", [])
        ]
    return decisions


def fit(
    model: Jean, decisions: list[Decision], epochs: int = 1, lr: float = 2e-6
) -> None:
    """Update all weights with candidate cross entropy; shuffle choice positions."""
    rng = random.Random(42)
    model.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        rng.shuffle(decisions)
        total = 0.0
        for decision in decisions:
            if decision["type"] == "choice":
                items = list(decision["choices"].items())
                rng.shuffle(items)
                decision = {**decision, "choices": dict(items)}
            logits = model(decision)
            target = torch.tensor(
                [list(decision["choices"]).index(decision["target"])],
                device=logits.device,
            )
            loss = torch.nn.functional.cross_entropy(logits[None], target)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            total += loss.item()
        print(f"epoch={epoch + 1} loss={total / len(decisions):.4f}", flush=True)


def main() -> None:
    """Train, optionally check validation accuracy, and save once at the end."""
    parser = argparse.ArgumentParser(description="Train Jean on a small JSONL dataset")
    parser.add_argument("data")
    parser.add_argument("output")
    parser.add_argument("--validation")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0 or Path(args.output).exists():
        parser.error("Use positive training settings and a fresh output directory")
    decisions = read_data(args.data)
    validation = read_data(args.validation) if args.validation else []
    torch.manual_seed(42)
    model = Jean.load(device=args.device)
    fit(model, decisions, args.epochs, args.lr)
    if validation:
        correct = 0
        for decision in validation:
            probabilities = model.predict(decision)["probabilities"]
            correct += max(probabilities, key=probabilities.get) == decision["target"]
        accuracy = correct / len(validation)
        print(f"validation_accuracy={accuracy:.3f}")
    model.save(args.output)


if __name__ == "__main__":
    main()
