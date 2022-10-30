import pandas as pd
import csv
df= pd.read_csv("brightest_stars_formatted.csv")
print(df.shape)
del df["luminosity"]
print(df.shape)
print(list(df))
df.to_csv("project130.csv")