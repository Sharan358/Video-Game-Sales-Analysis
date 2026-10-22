import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

#Correlation and Covariance Analysis
print("\nCorrelation:")
print(df.corr(numeric_only=True))

print("\nCovariance:")
print(df.cov(numeric_only=True))

print("Global Sales Skewness:", df['Global_Sales'].skew())

print("Global Sales Kurtosis:", df['Global_Sales'].kurt())

Q1 = df['Global_Sales'].quantile(0.25)
Q3 = df['Global_Sales'].quantile(0.75)

IQR = Q3 - Q1

outliers = df[
    (df['Global_Sales'] < Q1 - 1.5 * IQR) |
    (df['Global_Sales'] > Q3 + 1.5 * IQR)
]

print("Number of outliers:", len(outliers))

##------EDA-------##

#



# Video Game Distribution by Genre
genre_counts = df['Genre'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(
    genre_counts,
    labels=genre_counts.index,
    autopct='%1.1f%%',
)

plt.title('Video Game Distribution by Genre')
plt.tight_layout()
plt.show()

#
plt.figure(figsize=(8, 4))

sns.histplot(
    df['Global_Sales'].dropna(),
    bins=25,
    kde=True,
    color='purple',
    edgecolor='black'
)

plt.title('Distribution of Global Video Game Sales')
plt.xlabel('Global Sales (Millions)')
plt.ylabel('Frequency')
plt.show()


# 
plt.figure(figsize=(6, 4))

sns.boxplot(
    x=df['Global_Sales'],
    color='skyblue'
)

plt.title('Boxplot of Global Video Game Sales')
plt.xlabel('Global Sales (Millions)')
plt.show()


#
plt.figure(figsize=(10, 7))
corr = df.drop(columns='Rank').select_dtypes(include='number').corr()
sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5
)
plt.title('Correlation Heatmap of Video Game Sales')
plt.show()

#
year_sales = df.groupby('Year')['Global_Sales'].sum().reset_index()
plt.figure(figsize=(9, 5))

sns.scatterplot(
    data=year_sales,
    x='Year',
    y='Global_Sales',
    color='dodgerblue'
)

plt.title('Year vs Global Video Game Sales')
plt.xlabel('Release Year')
plt.ylabel('Global Sales (Millions)')
plt.show()
