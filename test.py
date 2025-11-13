from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DebertaV2Tokenizer, DebertaV2TokenizerFast
from scipy.stats import pearsonr, spearmanr
import pandas as pd
from scipy.special import expit
import matplotlib.pyplot as plt
import warnings
import time

class CorrelationDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128):
        self.max_length = max_length
        self.tokenizer = tokenizer


        df = pd.read_csv(csv_file)
        self.questions = df["prompt"].astype(str).tolist()
        self.responses = df["response"].astype(str).tolist()
        self.scores = torch.tensor(df["score"].values, dtype=torch.float)

        q_enc = self.tokenizer(
                self.questions,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
        )
        r_enc = self.tokenizer(
                self.responses,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )

        self.encodings = {
                "question_input": q_enc,
                "response_input": r_enc,
                "score": self.scores,
                "question_text": self.questions,
            }


    def __len__(self):
        return len(self.encodings["score"])

    def __getitem__(self, idx):
        return {
            "question_input": {k: v[idx] for k, v in self.encodings["question_input"].items()},
            "response_input": {k: v[idx] for k, v in self.encodings["response_input"].items()},
            "score": self.encodings["score"][idx],
            "question_text": self.encodings["question_text"][idx],
        }


def computeCorrelation(model, csv_path, batch_size, tokenizer_name, max_length=128, ho=False):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    print("Started loading dataset")

    dataset = CorrelationDataset(csv_path, tokenizer, max_length=max_length)

    print("Finished loading dataset")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )
    warnings.filterwarnings("ignore", message="The current process just got forked")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    preds, targets, questions = [], [], []
    total_examples = len(dataset)
    total_inference_time = 0.0
    iteration = 0

    with torch.no_grad():
        for batch in dataloader:
            q_input = {k: v.to(device, non_blocking=True) for k, v in batch["question_input"].items()}
            r_input = {k: v.to(device, non_blocking=True) for k, v in batch["response_input"].items()}
            score_true = batch["score"].to(device, non_blocking=True)

            start_time = time.perf_counter()
            score_pred = model.model(q_input, r_input)
            end_time = time.perf_counter()
            total_inference_time += (end_time - start_time)


            preds.append(score_pred.cpu())
            targets.append(score_true.cpu())
            questions.extend(batch["question_text"])
            iteration += 1
            # if (iteration % 1) == 0:
            #     print(iteration, " out of ", len(dataloader))

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    df = pd.DataFrame({
        "question": questions,
        "pred": preds,
        "target": targets,
    })

    pearson_corr = pearsonr(df["pred"], df["target"])[0]

    if not ho:
        df.to_csv("pred_vs_actual.csv", index=False)


    print(f"Pearson correlation: {pearson_corr:.4f}")
    avg_time_per_example = total_inference_time / total_examples
    print(f"Avg time for each example: {avg_time_per_example:.4f}")



    plt.figure(figsize=(10, 18))
    plt.xlabel("Actual scores")
    plt.ylabel("Predicted scores (per-prompt normalized)")
    plt.title(f"Predicted vs Actual (r={pearson_corr:.2f})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pred_vs_actual_perprompt.png", dpi=150, bbox_inches="tight")
    plt.close()

    if (not ho):
        df.to_csv("pred_vs_actual_perprompt.csv", index=False)
        plt.savefig("pred_vs_actual_perprompt.png", dpi=150, bbox_inches="tight")

    else:
        df.to_csv("pred_vs_actual_perprompt_ho.csv", index=False)
        plt.savefig("pred_vs_actual_perprompt_ho.png", dpi=150, bbox_inches="tight")

    return pearson_corr