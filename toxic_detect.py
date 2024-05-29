import bentoml
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
from typing import List


MODEL_PATH = "martin-ha/toxic-comment-model"

@bentoml.service(
    traffic={
        "timeout": 10,
        "concurrency": 500,
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
    },
)
class ToxicClassifier:
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        self.pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)

    @bentoml.api(batchable=True)
    def classify(self, texts: List[str]):
        return self.pipeline(texts)
