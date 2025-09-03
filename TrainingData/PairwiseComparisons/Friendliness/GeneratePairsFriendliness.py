import csv
import pandas as pd
from itertools import combinations

friendliness = "C:\\Users\\samgr\\PycharmProjects\\CreativityLLM\\TrainingData\\DataByPrompt\\Friendliness.csv"
friendlinessUncut = "C:\\Users\\samgr\\PycharmProjects\\CreativityLLM\\TrainingData\\PairwiseComparisons\\FriendlinessPairs.csv"
friendlinessCut = "C:\\Users\\samgr\\PycharmProjects\\CreativityLLM\\TrainingData\\PairwiseComparisons\\FriendlinessPairsCut.csv"

data_2D = []
with open(friendliness) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        data_2D.append(row)

data_2D_cut = []
with open(friendliness) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        if row[12] != "cut":
            data_2D_cut.append(row)
print(len(data_2D))
print(len(data_2D_cut))
with open(friendlinessUncut, 'w', newline="") as f:
    writer = csv.writer(f)
    for row1 in data_2D:
        for row2 in data_2D:
            if row1[6] == row2[6]:
                continue
            else:
                try:
                    diff = float(row1[8]) - float(row2[8])
                    writer.writerow([row1[6], row2[6], diff])
                except (ValueError, IndexError) as e:
                    print(f"Error processing a row: {e}")
                    continue


with open(friendlinessCut, 'w', newline="") as f:
    writer = csv.writer(f)


    for row1 in data_2D_cut:
        for row2 in data_2D_cut:
            if row1[6] == row2[6]:
                continue
            else:
                try:
                    diff = float(row1[8]) - float(row2[8])
                    writer.writerow([row1[6], row2[6], diff])
                except (ValueError, IndexError) as e:
                    print(f"Error processing a row: {e}")
                    continue