import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """
    Load the credit score data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pandas.DataFrame: The loaded data.
    """
    real_estate_data = pd.read_csv(file_path)
    return real_estate_data

# Load the credit score data
real_estate_data = load_data("credit_score.csv")

# Display the first few rows of the data
real_estate_data_head = real_estate_data.head()

# Display information about the data
data_info = real_estate_data.info()

# Print the first few rows of the data
print(real_estate_data_head)

# Print information about the data
print(data_info)

# Check for missing values in the data
print(real_estate_data.isnull().sum())

# Calculate descriptive statistics of the data
descriptive_stats = real_estate_data.describe()

# Print descriptive statistics
print(descriptive_stats)

# Set the style for the plots
sns.set_style("whitegrid")

# Create subplots for histograms of the data
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))
fig.suptitle('Histograms of Real Estate Data', fontsize=16)

# Specify the columns for the histograms
cols = ['House age', 'Distance to the nearest MRT station', 'Number of convenience stores',
        'Latitude', 'Longitude', 'House price of unit area']

# Plot histograms for each column
for i, col in enumerate(cols):
    sns.histplot(real_estate_data[col], kde=True, ax=axes[i//2, i%2])
    axes[i//2, i%2].set_title(col)
    axes[i//2, i%2].set_xlabel('')
    axes[i//2, i%2].set_ylabel('')

# Adjust the layout of the subplots
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Create subplots for scatter plots with house price of unit area
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
fig.suptitle('Scatter Plots with House Price of Unit Area', fontsize=16)

# Plot scatter plots for each feature against house price of unit area
sns.scatterplot(data=real_estate_data, x='House age', y='House price of unit area', ax=axes[0, 0])
sns.scatterplot(data=real_estate_data, x='Distance to the nearest MRT station', y='House price of unit area', ax=axes[0, 1])
sns.scatterplot(data=real_estate_data, x='Number of convenience stores', y='House price of unit area', ax=axes[1, 0])
sns.scatterplot(data=real_estate_data, x='Latitude', y='House price of unit area', ax=axes[1, 1])

# Adjust the layout of the subplots
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Calculate the correlation matrix
correlation_matrix = real_estate_data.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

# Print the correlation matrix
print(correlation_matrix)

# Select the features and target variable
features = ['Distance to the nearest MRT station', 'Number of convenience stores', 'Latitude', 'Longitude']
target = 'House price of unit area'

# Split the data into training and test sets
X = real_estate_data[features]
y = real_estate_data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_lr = model.predict(X_test)

# Plot the actual vs. predicted house prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted House Prices')
plt.show()
