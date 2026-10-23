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

# Global Sales Analysis by Year

year_sales = df.groupby('Year')['Global_Sales'].sum().reset_index()

year_sales = year_sales.sort_values('Year')

print("\nGlobal Sales by Year:")
print(year_sales)

print("\nYear with Highest Global Sales:")
print(year_sales.loc[year_sales['Global_Sales'].idxmax()])

# Number of Games Released Each Year

year_games = df.groupby('Year')['Name'].count().reset_index()

year_games.columns = ['Year', 'Number_of_Games']

print("\nGames Released by Year:")
print(year_games)

print("\nYear with Highest Game Releases:")
print(year_games.loc[year_games['Number_of_Games'].idxmax()])


# Year-wise Global Sales Trend

plt.figure(figsize=(12, 5))

sns.lineplot(
    data=year_sales,
    x='Year',
    y='Global_Sales',
    marker='o'
)

plt.title('Global Video Game Sales Trends Over the Years')
plt.xlabel('Release Year')
plt.ylabel('Global Sales (Millions)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Global Sales by Platform

platform_sales = df.groupby('Platform')['Global_Sales'].sum()

platform_sales = platform_sales.sort_values(ascending=False)
print("\nOBJECTIVE 1: GAMING PLATFORM PERFORMANCE")
print("\nGlobal Sales by Platform:")
print(platform_sales)

print("\nTop Performing Platform:")
print(platform_sales.head(1))

# Number of Sales by Genre

genre_sales_count = df.groupby('Genre')['Global_Sales'].sum()
genre_sales_count = genre_sales_count.sort_values(ascending=False)

print("\nOBJECTIVE 2: VIDEO GAME GENRE ANALYSIS")

print("\nTotal Global Sales by Genre:")
print(genre_sales_count)

print("\nGenre with Highest Sales:")
print(genre_sales_count.head(1))

#Sales Distribution Across Different Regions
regional_sales = df[
    ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
].sum()

regional_sales = regional_sales.sort_values(ascending=False)

print("\nOBJECTIVE 3: REGIONAL SALES ANALYSIS")

print("\n3. Total Sales by Region:")
print(regional_sales)
print("\nRegion with Highest Sales:")
print(regional_sales.head(1))

#Top Publishers by Global Sales
publisher_sales = df.groupby('Publisher')['Global_Sales'].sum()
publisher_sales = publisher_sales.sort_values(ascending=False)

print("\nOBJECTIVE 4: PUBLISHER PERFORMANCE ANALYSIS")

print("\n4. Global Sales by Publisher:")
print(publisher_sales.head(10))
print("\nTop Performing Publisher:")
print(publisher_sales.head(1))

#Top 10 Best-Selling Video Games
top_games = df.groupby('Name')['Global_Sales'].sum()
top_games = top_games.sort_values(ascending=False).head(10)

print("\nOBJECTIVE 5: BEST-SELLING VIDEO GAME ANALYSIS")
print("\n5. Top 10 Best-Selling Video Games:")
print(top_games)

print("\nBest-Selling Video Game:")
print(top_games.head(1))

plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_games.values,
    y=top_games.index,
    color='skyblue'
)

plt.title('Top 10 Best-Selling Video Games Worldwide')
plt.xlabel('Global Sales (Millions)')
plt.ylabel('Video Game Name')

plt.tight_layout()
plt.show()