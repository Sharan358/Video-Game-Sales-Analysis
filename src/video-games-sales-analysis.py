import pandas as pd

df = pd.read_csv('data/raw/vgsales.csv')

##------Data Pre-Processing----##
print(df.info())
print(df.describe())
print(df.head(10))

print("\nBefore Cleaning")
print(df.isnull().sum())

##------Data Cleaning-------##

df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')

df['Year'] = df['Year'].fillna(df['Year'].mode()[0])

df['Publisher'] = df['Publisher'].fillna('Unknown')

print("\nAfter Cleaning")
print(df.isnull().sum())
print(df.info())

df.to_csv('data/processed/vgsales_cleaned.csv', index=False)

