import bentoml
from typing import List


MODEL_PATH = "martin-ha/toxic-comment-model"

@bentoml.service(
    traffic={
        "concurrency": 400,
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-tesla-t4",
    },
)
class ToxicClassifier:
    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
        self.pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)


    @bentoml.api(batchable=True)
    def classify(self, texts: List[str]):
        return self.pipeline(texts)
