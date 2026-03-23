"""Step 1: verify LIAR-style BERT checkpoint loads and discriminates."""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from transformers import BertModel, BertTokenizer


class BERTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 2)

    def forward(self, ids, mask):
        out = self.bert(input_ids=ids, attention_mask=mask)
        return self.classifier(self.dropout(out.pooler_output))


def main():
    model = BERTClassifier()
    model.load_state_dict(torch.load("models/bert_classifier.pt", map_location="cpu"))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    tests = [
        ("Scientists confirm peer reviewed findings in Nature.", "credible"),
        ("SHOCKING cover-up exposed by insiders today!", "misinfo"),
        ("University study validates clinical trial results.", "credible"),
        ("Miracle cure doctors do not want you to know!", "misinfo"),
        ("Official government data confirms steady growth.", "credible"),
        ("Illuminati controls all world governments confirmed!", "misinfo"),
    ]
    passed = 0
    with torch.no_grad():
        for text, expected in tests:
            enc = tokenizer(
                [text],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            out = model(enc["input_ids"], enc["attention_mask"])
            p = torch.softmax(out, dim=1)[0][1].item()
            ok = (expected == "credible" and p < 0.4) or (
                expected == "misinfo" and p > 0.6
            )
            if ok:
                passed += 1
            print(
                "%s [%s] P=%.4f  %s"
                % ("OK  " if ok else "FAIL", expected, p, text[:50])
            )
    print("BERT local verify: %d/6" % passed)


if __name__ == "__main__":
    main()
