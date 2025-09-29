from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from scipy.stats import pearsonr
import pandas as pd

class CorrelationDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name="bert-base-uncased", max_length=128):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        prompt = str(row[0])
        response = str(row[1])
        label = float(row[2])

        # Encode the pair for BERT-style models
        inputs = self.tokenizer(
            prompt,
            response,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Remove batch dimension
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item["score"] = torch.tensor(label, dtype=torch.float)
        return item

def computeCorrelation(model, csv_path, batch_size, tokenizer_name="bert-base-uncased", max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = CorrelationDataset(csv_path, tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle = False)

    model.eval()
    preds = []
    targets = []

    device = "cuda"
    model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            q_input = {k: v.to(device) for k, v in batch["question_input"].items()}
            r_input = {k: v.to(device) for k, v in batch["response_input"].items()}

            scores = model.model(q_input, r_input)  # PolyEncoder returns single score
            preds.append(scores.cpu())
            targets.append(batch["score"].cpu())

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    pearson_corr = pearsonr(preds, targets)[0]
    print(f"Pearson correlation: {pearson_corr:.4f}")
    return pearson_corr