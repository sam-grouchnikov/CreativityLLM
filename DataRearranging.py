import pandas as pd

df = pd.read_csv("C:\\Users\\samgr\\PycharmProjects\\CreativityLLM\\HeldOutTest.csv")

df = df[["prompt","response","label"]]

df.to_csv("C:\\Users\\samgr\\PycharmProjects\\CreativityLLM\\HeldOutTest.csv")