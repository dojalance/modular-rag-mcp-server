"""Serve the frozen Cross-Encoder from the existing CUDA PyTorch environment."""

import json
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(sys.argv[1], local_files_only=True).to("cuda").eval()
    print(json.dumps({"device": torch.cuda.get_device_name(), "torch": torch.__version__}), flush=True)
    with torch.inference_mode():
        for line in sys.stdin:
            pairs = json.loads(line)
            queries, passages = zip(*pairs, strict=True)
            inputs = tokenizer(list(queries), list(passages), padding=True, truncation=True, max_length=512,
                               return_tensors="pt").to("cuda")
            scores = model(**inputs).logits.reshape(-1).detach().cpu().tolist()
            print(json.dumps(scores), flush=True)


if __name__ == "__main__":
    main()
