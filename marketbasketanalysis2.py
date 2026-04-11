# Step 1 - Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

# Step 2 - Create Dataset
transactions = [
    ['bread', 'butter', 'milk'],
    ['bread', 'butter'],
    ['bread', 'milk', 'eggs'],
    ['butter', 'milk', 'eggs'],
    ['bread', 'butter', 'milk', 'eggs'],
    ['bread', 'chips', 'cold drink'],
    ['chips', 'cold drink', 'eggs'],
    ['bread', 'chips', 'cold drink', 'milk'],
    ['butter', 'eggs', 'milk'],
    ['bread', 'butter', 'eggs', 'milk'],
    ['chips', 'cold drink'],
    ['bread', 'milk', 'butter', 'chips'],
    ['eggs', 'milk', 'butter'],
    ['bread', 'eggs', 'cold drink'],
    ['chips', 'cold drink', 'milk']
]

# Step 3 - Prepare Data
te = TransactionEncoder()
te_array = te.fit_transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)
print("Dataset:")
print(df.head())

# Step 4 - Apply Apriori Algorithm
frequent_items = apriori(df, 
                         min_support=0.3, 
                         use_colnames=True)
print("\nFrequent Items:")
print(frequent_items)

# Step 5 - Generate Association Rules
rules = association_rules(frequent_items, 
                         metric="confidence",
                         min_threshold=0.6)
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 
             'support', 'confidence', 'lift']])

# Step 6 - Visualize Top Rules
plt.figure(figsize=(10,6))
sns.scatterplot(x='support', 
                y='confidence',
                size='lift',
                data=rules,
                sizes=(50,500))
plt.title('Market Basket Analysis - Association Rules')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.show()

# Step 7 - Best Rules
print("\nTop Association Rules by Lift:")
top_rules = rules.sort_values('lift', 
                               ascending=False).head(5)
print(top_rules[['antecedents', 'consequents', 
                 'support', 'confidence', 'lift']])