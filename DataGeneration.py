import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split
import csv

# # Load SCTT file
# df = pd.read_csv("C:\\Users\\samgr\\PycharmProjects\\CreativityLLM\\Filtered.csv", header=0)
#
# # Clean column names and select relevant columns
# df.columns = df.columns.str.strip()
# df = df[['prompt', 'response', 'se', 'cut']]  # include 'cut' column
# df['se'] = df['se'].astype(float)
#
# # Keep only rows where cut == "keep"
# df = df[df['cut'].str.lower() == 'keep']
#
# train_individual = []
# test_individual = []
# line_count = 0
#
# # Open CSV for train/val pairs (write incrementally)
# train_file = "C:\\Users\\samgr\\PycharmProjects\\CreativityLLM\\TrainData.csv"
# test_file = "C:\\Users\\samgr\\PycharmProjects\\CreativityLLM\\TestData.csv"
# with open(train_file, "w", encoding="utf-8", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["prompt", "response1", "response2", "label"])  # header
#
#     # Process each prompt individually
#     for prompt, group in df.groupby('prompt'):
#         # Step 1: Split 80% train/val, 20% test
#         train_val, test = train_test_split(group, test_size=0.2, random_state=42)
#
#         # Step 2: Generate all possible pairs (combinations) and also the flipped pairs
#         for r1, r2 in combinations(train_val.itertuples(index=False), 2):
#             label = r1.se - r2.se
#             # Original order
#             writer.writerow([prompt, r1.response, r2.response, label])
#             line_count += 1
#             # Flipped order
#             writer.writerow([prompt, r2.response, r1.response, -label])
#             line_count += 1
#
#         # Step 3: Keep individual responses for test
#         for r in test.itertuples(index=False):
#             test_individual.append({
#                 "prompt": prompt,
#                 "response": r.response,
#                 "score": r.se
#             })
#
#
#
# # Convert test responses to DataFrame and save
# test_df = pd.DataFrame(test_individual)
# test_df.to_csv(test_file, index=False)
#
# print(f"Number of train/val pairs written (bi-directional): {line_count}")
# print(f"Number of test individual responses: {len(test_df)}")


original = "C:\\Users\\samgr\\PycharmProjects\\CreativityLLM\\TrainingData\\Filtered.csv"
train_file = "C:\\Users\\samgr\\PycharmProjects\\CreativityLLM\\TrainingData\\TrainData.csv"
test_file = "C:\\Users\\samgr\\PycharmProjects\\CreativityLLM\\TrainingData\\TestData.csv"

# Load the original CSV
df = pd.read_csv(original)

# Split into 80% train and 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Save the results
train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)

print(f"Training data saved to {train_file} ({len(train_df)} rows)")
print(f"Test data saved to {test_file} ({len(test_df)} rows)")