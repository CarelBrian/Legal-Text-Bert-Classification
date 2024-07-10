import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel
import pandas as pd
import numpy as np
import gradio as gr


class CustomBert(nn.Module):
    def __init__(self, model_name_or_path="bert-base-uncased", n_classes=10):
        super(CustomBert, self).__init__()
        self.bert_pretained = BertModel.from_pretrained(model_name_or_path)
        self.classifier = nn.Linear(self.bert_pretained.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        x = self.bert_pretained(input_ids=input_ids, attention_mask=attention_mask)

        x = self.classifier(x.pooler_output)

        return x


model = CustomBert()
model.load_state_dict(torch.load("my_legal_text_classification_bert.pth"))  # load saved model


def classifier_fn(text: str):
    labels = {
        0: "cited", 
        1: "applied", 
        2: "followed", 
        3: "referred to", 
        4: "related", 
        5: "considered", 
        6: "discussed", 
        7: "distinguished", 
        8: "affirmed", 
        9: "approved"
    }

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer(
        text, padding="max_length", max_length=250, truncation=True, return_tensors="pt"
    )

    output = model(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
    )

    _, pred = output.max(1)

    return labels[pred.item()]


demo = gr.Interface(
    fn=classifier_fn,
    inputs=["text"],
    outputs=["text"],
)

demo.launch()
