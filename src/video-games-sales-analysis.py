import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

# LOAD DATASET
# Load the video games sales dataset from the local CSV file
df = pd.read_csv('data/raw/vgsales.csv')

# DATA PRE-PROCESSING

print(df.info())
print(df.describe())
print(df.head(10))

# Display the number of missing values in each column before cleaning
print("\nBefore Cleaning")
print(df.isnull().sum())

# DATA CLEANING

# Convert the release year to a numeric data type
df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
 
# Replace missing release years with the most frequent year (mode)
df['Year'] = df['Year'].fillna(df['Year'].mode()[0])
 
# Replace missing publishers with the label 'Unknown'
df['Publisher'] = df['Publisher'].fillna('Unknown')

# Display the number of remaining missing values after cleaning
print("\nAfter Cleaning")
print(df.isnull().sum())
print(df.info())

# Save the cleaned dataset to the processed data folder
df.to_csv('data/processed/vgsales_cleaned.csv', index=False)

# CORRELATION AND COVARIANCE ANALYSIS

# Calculate correlations between numerical columns
# Correlation measures the strength and direction of linear relationships
print("\nCorrelation:")
print(df.corr(numeric_only=True))

# Calculate covariance between numerical columns
# Covariance indicates whether two variables tend to vary together
print("\nCovariance:")
print(df.cov(numeric_only=True))

# SKEWNESS AND KURTOSIS ANALYSIS

# Calculate global sales skewness to understand distribution asymmetry
print("Global Sales Skewness:", df['Global_Sales'].skew())

# Calculate global sales kurtosis to examine distribution tail characteristics
print("Global Sales Kurtosis:", df['Global_Sales'].kurt())

# OUTLIER DETECTION USING THE IQR METHOD

Q1 = df['Global_Sales'].quantile(0.25)
Q3 = df['Global_Sales'].quantile(0.75)

IQR = Q3 - Q1

outliers = df[
    (df['Global_Sales'] < Q1 - 1.5 * IQR) |
    (df['Global_Sales'] > Q3 + 1.5 * IQR)
]

print("Number of outliers:", len(outliers))

 
# EXPLORATORY DATA ANALYSIS (EDA)

# VISUALIZATION: VIDEO GAME DISTRIBUTION BY GENRE
 
# Count the number of games in each genre
genre_counts = df['Genre'].value_counts()
 
# Plot a pie chart to show the share of each genre
plt.figure(figsize=(6,6))
plt.pie(
    genre_counts,
    labels=genre_counts.index,
    autopct='%1.1f%%',
)
 
plt.title('Video Game Distribution by Genre')
plt.tight_layout()
plt.show()

# VISUALIZATION: DISTRIBUTION OF GLOBAL SALES
 
# Plot a histogram with a KDE curve to visualize sales distribution
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

# VISUALIZATION: GLOBAL SALES BOXPLOT
 
# Use a boxplot to visualize sales spread and potential outliers
plt.figure(figsize=(6, 4))
sns.boxplot(
    x=df['Global_Sales'],
    color='skyblue'
)
plt.title('Boxplot of Global Video Game Sales')
plt.xlabel('Global Sales (Millions)')
plt.show()

# VISUALIZATION: CORRELATION HEATMAP
 
# Select the numerical features, excluding the Rank column
corr = df.drop(columns='Rank').select_dtypes(include='number').corr()
 
# Visualize correlations using a heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5
)
plt.title('Correlation Heatmap of Video Game Sales')
plt.show()

# VISUALIZATION: RELEASE YEAR VS. GLOBAL SALES
 
# Calculate the total global sales for each release year
year_sales = df.groupby('Year')['Global_Sales'].sum().reset_index()
 
# Plot the relationship between release year and total sales
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


# GLOBAL SALES ANALYSIS BY YEAR

year_sales = df.groupby('Year')['Global_Sales'].sum().reset_index()

year_sales = year_sales.sort_values('Year')

print("\nGlobal Sales by Year:")
print(year_sales)

print("\nYear with Highest Global Sales:")
print(year_sales.loc[year_sales['Global_Sales'].idxmax()])

# NUMBER OF GAMES RELEASED EACH YEAR

year_games = df.groupby('Year')['Name'].count().reset_index()

year_games.columns = ['Year', 'Number_of_Games']

print("\nGames Released by Year:")
print(year_games)

print("\nYear with Highest Game Releases:")
print(year_games.loc[year_games['Number_of_Games'].idxmax()])


# VISUALIZATION: YEAR-WISE GLOBAL SALES TREND

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


# FIVE OBJECTIVES OF THE VIDEO GAMES SALES PROJECT


# Global Sales by Platform

# Objective 1: Analyze Gaming Platform Performance
# Calculate the total global sales for each platform

platform_sales = df.groupby('Platform')['Global_Sales'].sum()

platform_sales = platform_sales.sort_values(ascending=False)
print("\nOBJECTIVE 1: GAMING PLATFORM PERFORMANCE")
print("\nGlobal Sales by Platform:")
print(platform_sales)

print("\nTop Performing Platform:")
print(platform_sales.head(1))

# Objective 2: Study Video Game Genre Patterns
# Calculate the total global sales for each genre

genre_sales_count = df.groupby('Genre')['Global_Sales'].sum()
genre_sales_count = genre_sales_count.sort_values(ascending=False)

print("\nOBJECTIVE 2: VIDEO GAME GENRE ANALYSIS")

print("\nTotal Global Sales by Genre:")
print(genre_sales_count)

print("\nGenre with Highest Sales:")
print(genre_sales_count.head(1))

# Objective 3: Investigate Sales Distribution Across Regions
# Calculate the total sales in each region

regional_sales = df[
    ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
].sum()

regional_sales = regional_sales.sort_values(ascending=False)

print("\nOBJECTIVE 3: REGIONAL SALES ANALYSIS")

print("\n3. Total Sales by Region:")
print(regional_sales)
print("\nRegion with Highest Sales:")
print(regional_sales.head(1))

# Objective 4: Examine Top Publishers by Global Sales
# Calculate the total global sales for each publisher

publisher_sales = df.groupby('Publisher')['Global_Sales'].sum()
publisher_sales = publisher_sales.sort_values(ascending=False)

print("\nOBJECTIVE 4: PUBLISHER PERFORMANCE ANALYSIS")

print("\n4. Global Sales by Publisher:")
print(publisher_sales.head(10))
print("\nTop Performing Publisher:")
print(publisher_sales.head(1))

# Objective 5: Identify the Top 10 Best-Selling Video Games
# Calculate the total global sales for each game and keep the top ten

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
# Plot a horizontal bar chart of the top 10 best-selling games
plt.title('Top 10 Best-Selling Video Games Worldwide')
plt.xlabel('Global Sales (Millions)')
plt.ylabel('Video Game Name')

plt.tight_layout()
plt.show()

# Create sales categories using the median as the cutoff between low and high
median_sales = df['Global_Sales'].median()
print("Median Global Sales:", median_sales)

# Games at or below the median are Low Sales, games above it are High Sales
df['Sales_Category'] = pd.cut(
    df['Global_Sales'],
    bins=[-0.01, median_sales, 100],
    labels=['Low Sales', 'High Sales']
)

print(df['Sales_Category'].value_counts())

# Select input features and target variable
X = df[['Platform', 'Genre', 'Year', 'Publisher']].copy()
Y = df['Sales_Category']

# Encode categorical features into numerical value
le_platform = LabelEncoder()
le_genre = LabelEncoder()
le_publisher = LabelEncoder()

X['Platform'] = le_platform.fit_transform(X['Platform'])
X['Genre'] = le_genre.fit_transform(X['Genre'])
X['Publisher'] = le_publisher.fit_transform(X['Publisher'])

# Split data into training and testing sets
# Stratify keeps the Low and High class proportions the same in both sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# CLASSIFICATION MODELS

# CLASSIFICATION MODEL - 1 : Logistic Regression
# Create and train the Logistic Regression model
model1 = LogisticRegression(max_iter=1000, class_weight='balanced')
model1.fit(X_train, Y_train)
# Predict sales categories
y_pred1 = model1.predict(X_test)
# Evaluate the model
print("\nEvaluation parameters for Logistic Regression model")
print("Accuracy:", accuracy_score(Y_test, y_pred1))
print("Precision:", precision_score(Y_test, y_pred1, average='macro', zero_division=0))
print("Recall:", recall_score(Y_test, y_pred1, average='macro', zero_division=0))
print("F1 Score:", f1_score(Y_test, y_pred1, average='macro', zero_division=0))
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, y_pred1))
print("\nClassification Report:")
print(classification_report(Y_test, y_pred1, zero_division=0))

# CLASSIFICATION MODEL - 2 : KNN
# Create and train the KNN model
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train, Y_train)
# Predict sales categories
y_pred2 = model2.predict(X_test)
# Evaluate the model
print("\nEvaluation parameters for KNN model")
print("Accuracy:", accuracy_score(Y_test, y_pred2))
print("Precision:", precision_score(Y_test, y_pred2, average='macro', zero_division=0))
print("Recall:", recall_score(Y_test, y_pred2, average='macro', zero_division=0))
print("F1 Score:", f1_score(Y_test, y_pred2, average='macro', zero_division=0))
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, y_pred2))
print("\nClassification Report:")
print(classification_report(Y_test, y_pred2, zero_division=0))

# CLASSIFICATION MODEL - 3 : Support Vector Machine
# Create and train the Support Vector Machine
model3 = SVC(class_weight='balanced')
model3.fit(X_train, Y_train)
# Predict sales categories
y_pred3 = model3.predict(X_test)
# Evaluate the model
print("\nEvaluation parameters for Support Vector Machine")
print("Accuracy:", accuracy_score(Y_test, y_pred3))
print("Precision:", precision_score(Y_test, y_pred3, average='macro', zero_division=0))
print("Recall:", recall_score(Y_test, y_pred3, average='macro', zero_division=0))
print("F1 Score:", f1_score(Y_test, y_pred3, average='macro', zero_division=0))
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, y_pred3))
print("\nClassification Report:")
print(classification_report(Y_test, y_pred3, zero_division=0))
