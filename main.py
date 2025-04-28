import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Load the dataset
Data = pd.read_csv("shopping_trends.csv")   

# ✅ Display the first few rows
print(Data.head())

# ✅ Basic info
Data.info()

# ✅ Descriptive statistics for numerical columns
print(Data.describe())

# ✅ Descriptive statistics for categorical columns
print(Data.describe(include="object"))

# ✅ Value counts for 'Age'
print(Data['Age'].value_counts())

# ✅ Define categorize_age function
def categorize_age(age):
    if 18 <= age < 30:
        return 'Young'
    elif 30 <= age < 50:
        return 'Mid-age'
    elif 50 <= age < 70:
        return 'Old'
    else:
        return 'Other'

# ✅ Create Age Category
Data['Age_category'] = Data['Age'].apply(categorize_age)

# ✅ Plot Age Category distribution
sns.countplot(data=Data, x='Age_category')
plt.title('Age Category Distribution')
plt.show()

# ✅ Items purchased by category
print(Data.groupby('Category')['Item Purchased'].value_counts())

# ✅ Gender purchase count
sns.countplot(x="Gender", data=Data)
plt.title("Between Males and Females, Who Buys More?")
plt.show()

# ✅ Total purchase amount by gender
amount = Data.groupby('Gender')['Purchase Amount (USD)'].sum()
print(amount)

# ✅ Payment method count and plot
count_method = Data['Payment Method'].value_counts().reset_index()
count_method.columns = ['Payment Method', 'Count']
plt.figure(figsize=(7,4))
sns.barplot(x='Payment Method', y='Count', data=count_method, palette='Blues')
plt.title('Count of Payment Methods', fontsize=14, fontweight='bold')
plt.xlabel('Payment Method', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.show()

# ✅ Top 8 items by season
print(Data.groupby('Season')['Item Purchased'].value_counts().sort_values(ascending=False).head(8))

# ✅ Average review rating by category
print(Data.groupby('Category')['Review Rating'].mean())

# ✅ Discount applied cross-tab
ax = pd.crosstab(Data["Discount Applied"], Data["Category"])
print(ax)

# ✅ Total purchase amount by season and category
print(pd.crosstab(Data['Season'], Data['Category'], values=Data['Purchase Amount (USD)'], aggfunc=np.sum))

# ✅ Subscription status count
print(Data['Subscription Status'].value_counts())

# ✅ Total purchase by subscription status
print(Data.groupby('Subscription Status')['Purchase Amount (USD)'].sum())

# ✅ Category purchases by Age Category
sns.countplot(data=Data, x='Category', hue='Age_category', palette=['#3498db', '#e74c3c', '#2ecc71'])
plt.title('Purchase Count by Category and Age Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ✅ Discounts applied
sns.countplot(x="Discount Applied", data=Data)
plt.title("How Many are Getting a Discount?")
plt.show()

# ✅ Shipping type count
Shipping_Type = Data['Shipping Type'].value_counts()
plt.figure(figsize=(7, 4))
sns.barplot(x=Shipping_Type.index, y=Shipping_Type.values, palette='Reds', linewidth=2)
plt.title('Type of Shipping')
plt.xlabel('Shipping Type')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# ✅ Frequency of purchases
print(Data['Frequency of Purchases'].value_counts())

# ✅ Promo code usage
print(Data.groupby('Promo Code Used')['Purchase Amount (USD)'].count())

# ✅ Top 10 locations by count
top_locations = Data['Location'].value_counts().head(10)
top_locations.plot(kind='bar')
plt.title('Top 10 Locations')
plt.ylabel('Count')
plt.show()

# ✅ Top 15 colors
Data['Color'].value_counts().head(15).plot(kind='bar', color='orange')
plt.title('Top 15 Colors Purchased')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ✅ Season vs Item Purchased
print(pd.crosstab(Data['Season'], Data['Item Purchased']).T)

# ✅ Size count
size_count = Data['Size'].value_counts()
plt.figure(figsize=(8,8))
plt.pie(size_count, labels=size_count.index, autopct='%0.0f%%', explode=(0,0,0,0))
plt.legend(size_count.index, loc=2)
plt.title('Size of the Purchased Item')
plt.tight_layout()
plt.show()

# ✅ Total purchase by size for Clothing
print(Data[Data['Category'] == 'Clothing'].groupby('Size')['Purchase Amount (USD)'].sum())

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns

# Define features and target variable
X = Data[['Age', 'Previous Purchases']]
y = Data['Purchase Amount (USD)']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Plot the predictions vs actual
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Purchase Amount (USD)')
plt.ylabel('Predicted Purchase Amount (USD)')
plt.title('Sales Prediction (Linear Regression)')
plt.show()
