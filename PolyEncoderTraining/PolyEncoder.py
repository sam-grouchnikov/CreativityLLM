import torch
import torch.nn as nn
from transformers import BertModel

class PolyEncoder(nn.Module):
    def __init__(self, model_name = "bert-base-uncased", code_count = 64):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.poly_codes = nn.Embedding(code_count, self.bert.config.hidden_size)
        self.code_count = code_count

    def encodeContext(self, input_ids, attention_mask):
        """
            Encode a batch of context sequences into poly-encoder context vectors.

            Args:
                input_ids: [b, T]  -- token IDs of context sequences (batch of sequences)
                attention_mask: [b, T] -- mask indicating valid tokens (1) vs padding (0)

            Returns:
                context_vectors: [b, m, h]  -- poly-encoded context embeddings
                    b = batch size
                    m = number of poly codes
                    h = hidden size
        """

        # 1. Encode context sequences with BERT
        ctx_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # ctx_outputs: [b, T, h]

        # 2. Get poly codes and expand across batch
        poly_codes = self.poly_codes.weight.unsqueeze(0).expand(ctx_outputs.size(0), -1, -1)
        # poly_codes: [b, m, h]

        # 3. Compute raw attention scores: [b, m, T]
        attention = torch.bmm(poly_codes, ctx_outputs.transpose(1, 2))

        # 4. Mask out padding tokens before softmax
        # attention_mask: [b, T] -> [b, 1, T] to broadcast across m poly codes
        mask = attention_mask.unsqueeze(1)  # [b, 1, T]
        attention = attention.masked_fill(mask == 0, float('-inf'))

        # 5. Apply softmax over tokens
        attention_weights = torch.softmax(attention, dim=-1)

        # 6. Weighted sum over context tokens
        context_vectors = torch.bmm(attention_weights, ctx_outputs)
        # context_vectors: [b, m, h]

        return context_vectors

    def encodeCandidate(self, input_ids, attention_mask):
        """
                Encode candidate sequence by taking the [CLS] token embedding
                Args:
                    input_ids: [b, T]
                    attention_mask: [b, T]
                Returns:
                    cand_vec: [b, h]
        """

        # Encode candidate sequences with BERT and get a single vector per candidate
        cand_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # cand_out: [batch_size, seq_len, hidden_size]

        # Take the first token ([CLS] token) embedding as the candidate vector
        cand_vec = cand_out[:, 0, :]
        # cand_vec: [batch_size, hidden_size]
        return cand_vec

    def forward(self, ctx_input, ctx_mask, cand_input, cand_mask):
        """
                Compute relevance score between context and candidate.
                Returns:
                    scores: [b]  -- max similarity across poly codes
        """

        ctx_vecs = self.encodeContext(ctx_input, ctx_mask) # [b, m, h]
        cand_vecs = self.encodeCandidate(cand_input, cand_mask) # [b, h]

        # [b, m, h] dot [b, h] -> [b, m]
        attn_scores = torch.bmm(ctx_vecs, cand_vecs.unsqueeze(-1)).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [b, m]

        # Weighted sum of poly codes
        ctx_final = torch.bmm(attn_weights.unsqueeze(1), ctx_vecs).squeeze(1)  # [b, h]

        # Final similarity score
        scores = torch.sum(ctx_final * cand_vecs, dim=-1)  # [b]

        return scores
