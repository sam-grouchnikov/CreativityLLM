import csv
import pandas as pd
from itertools import combinations

datasets = [
    "Friendliness", "Galaxy", "Holds", "Jungle", "MindReading",
    "OceanFloor", "Robots", "SenseOfHumor", "StarsDisappearing",
    "StudentsSinging", "WarmerLake", "WaterSunlight"
]

raw_data = [f"C:\\Users\\samgr\\PycharmProjects\\CreativityLLM\\TrainingData\\DataByPrompt\\{name}.csv" for name in datasets]

uncut_paths = [f"C:\\Users\\samgr\\PycharmProjects\\CreativityLLM\\TrainingData\\PairwiseComparisons\\{name}\\{name}Pairs.csv" for name in datasets]

cut_paths = [f"C:\\Users\\samgr\\PycharmProjects\\CreativityLLM\\TrainingData\\PairwiseComparisons\\{name}\\{name}PairsCut.csv" for name in datasets]

total_cut_pairs = 0
total_uncut_pairs = 0

print(len(raw_data))
for i in range(0, len(raw_data)):
    print(i)
    raw = raw_data[i]
    uncut = uncut_paths[i]
    cut = cut_paths[i]
    print(raw)
    data_2D = []
    with open(raw) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data_2D.append(row)

    data_2D_cut = []
    with open(raw) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if row[12] != "cut":
                data_2D_cut.append(row)
    cutPairs = 0
    uncutPairs = 0
    with open(uncut, 'w', newline="") as f:
        writer = csv.writer(f)
        for row1 in data_2D:
            for row2 in data_2D:
                if row1[6] == row2[6]:
                    continue
                else:
                    try:
                        diff = float(row1[8]) - float(row2[8])
                        writer.writerow([row1[6], row2[6], row1[8], row2[8]])
                        uncutPairs += 1
                    except (ValueError, IndexError) as e:
                        print(f"Error processing a row: {e}")
                        continue


    with open(cut, 'w', newline="") as f:
        writer = csv.writer(f)


        for row1 in data_2D_cut:
            for row2 in data_2D_cut:
                if row1[6] == row2[6]:
                    continue
                else:
                    try:
                        diff = float(row1[8]) - float(row2[8])
                        writer.writerow([row1[6], row2[6], row1[8], row2[8]])
                        cutPairs += 1
                    except (ValueError, IndexError) as e:
                        print(f"Error processing a row: {e}")
                        continue

    print("Cut responses: ", len(data_2D_cut))
    print("Uncut responses: ", len(data_2D))
    print("Cut pairs: ", cutPairs)
    print("Uncut pairs: ", uncutPairs)
    total_cut_pairs += cutPairs
    total_uncut_pairs += uncutPairs
    print()

print("Total cut pairs: ", total_cut_pairs)
print("Total uncut pairs: ", total_uncut_pairs)