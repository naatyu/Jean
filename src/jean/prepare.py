"""Download only 200 training and 50 validation decisions from Hugging Face BoolQ."""

import json
from pathlib import Path
from urllib.request import urlopen


def main() -> None:
    """Create a tiny local dataset in Jean's common format, without extra libraries."""
    output = Path("data/boolq")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("*.jsonl")):
        raise ValueError("Sample already exists; reuse it instead of downloading again")
    heldout = set()
    for split, count in [("validation", 50), ("train", 200)]:
        decisions = []
        for offset in range(0, count, 100):
            length = min(100, count - offset)
            url = f"https://datasets-server.huggingface.co/rows?dataset=google%2Fboolq&config=default&split={split}&offset={offset}&length={length}"
            with urlopen(url, timeout=30) as response:
                page = json.load(response)
            for item in page["rows"]:
                raw = item["row"]
                if raw["passage"] in heldout:
                    continue
                decisions.append(
                    {
                        "type": "noul",
                        "state": raw["passage"],
                        "question": raw["question"],
                        "choices": {"yes": "Yes", "no": "No"},
                        "target": "yes" if raw["answer"] else "no",
                    }
                )
        if split == "validation":
            heldout.update(decision["state"] for decision in decisions)
        path = output / f"{split}.jsonl"
        path.write_text(
            "".join(
                json.dumps(decision, ensure_ascii=False) + "\n"
                for decision in decisions
            )
        )
        print(f"{split}: {len(decisions)} examples, {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
