import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertModel
from torch.optim import Adam
from tqdm import tqdm
import pandas as pd
import numpy as np


class LegalTextClassificationDataset(Dataset):
    def __init__(
        self, 
        csv_file, 
        device, 
        model_name_or_path="bert-base-uncased", 
        max_length=250
    ):
        self.device = device
        self.df = pd.read_csv(csv_file)
        self.labels = self.df.case_outcome.unique()
        labels_dict = dict()
        for indx, l in enumerate(self.labels):
            labels_dict[l] = indx
        
        self.df['case_outcome'] = self.df['case_outcome'].map(labels_dict)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        
        case_text = self.df.case_text[index]
        if not isinstance(case_text, str):
            case_text = str(case_text)

        label_case = self.df.case_outcome[index]

        inputs = self.tokenizer(
            case_text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        labels = torch.tensor(label_case)

        return {
            "input_ids": inputs["input_ids"].squeeze(0).to(self.device),
            "attention_mask": inputs["attention_mask"].squeeze(0).to(self.device),
            "labels": labels.to(self.device)
        }



class CustomBert(nn.Module):
    def __init__(self, model_name_or_path="bert-base-uncased", n_classes=10):
        super(CustomBert, self).__init__()
        self.bert_pretained = BertModel.from_pretrained(model_name_or_path)
        self.classifier = nn.Linear(self.bert_pretained.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        x=self.bert_pretained(input_ids=input_ids, attention_mask=attention_mask)

        x=self.classifier(x.pooler_output)

        return x


def training_step(model, data_loader, loss_fn, optimizer):
    model.train()

    total_loss = 0

    for data in tqdm(data_loader, total=len(data_loader)):

        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        labels = data["labels"]

        output = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(data_loader.dataset)
    

def evaluation(model, test_dataloader, loss_fn):
    model.eval()

    correct_predictions= 0
    losses=[]

    for data in tqdm(test_dataloader, total=len(test_dataloader)):
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        labels = data["labels"]

        output = model(input_ids, attention_mask=attention_mask)

        _, pred = output.max(1)

        correct_predictions += torch.sum(pred==labels)

        loss = loss_fn(output, labels)

        losses.append(loss.item())

    return np.mean(losses) , correct_predictions /len(test_dataloader.dataset)





def main():

    print("Training ...")
    N_EPOCHS = 2 
    LR = 2e-5 
    BATCH_SIZE = 12

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_dataset = LegalTextClassificationDataset(csv_file="train_legal_text_classification.csv", device=device, max_length=100)
    test_dataset = LegalTextClassificationDataset(csv_file="test_legal_text_classification.csv", device=device, max_length=100)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

    model = CustomBert()
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=LR)

    for epoch in range(N_EPOCHS):
        loss_train = training_step(model, train_dataloader, loss_fn, optimizer)
        loss_eval, accuracy = evaluation(model, test_dataloader, loss_fn)

        print(
            f"Train Loss : {loss_train} | Eval Loss : {loss_eval} | Accuracy : {accuracy}"
        )
    
    # sauvegarde model
    torch.save(model.state_dict(), "my_legal_text_classification_bert.pth")



if __name__=="__main__":
    main()
