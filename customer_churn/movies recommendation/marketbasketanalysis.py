import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 1. SAMPLE DATASET
# (A list of lists where each inner list is a single transaction)
dataset = [
    ['Milk', 'Bread', 'Eggs'],
    ['Milk', 'Bread'],
    ['Milk', 'Diapers', 'Beer'],
    ['Bread', 'Eggs'],
    ['Milk', 'Bread', 'Eggs', 'Beer'],
    ['Bread', 'Eggs', 'Beer'],
    ['Milk', 'Diapers', 'Bread']
]

# 2. DATA ENCODING
# Transforming the list format into a boolean one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# 3. FIND FREQUENT ITEMSETS
# We set min_support to 0.3 (itemsets must appear in at least 30% of transactions)
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)

# 4. GENERATE ASSOCIATION RULES
# We use 'lift' as the primary metric to find strong relationships
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0, num_itemsets=len(frequent_itemsets))

# 5. FORMATTING FOR READABILITY
# Converting frozensets to lists so they are easier to read in the output
rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x))
rules["consequents"] = rules["consequents"].apply(lambda x: list(x))

# Sorting results by Lift to show the strongest associations first
result = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values('lift', ascending=False)

# 6. PRINT RESULTS
print("--- Frequent Itemsets ---")
print(frequent_itemsets)
print("\n--- Top Association Rules ---")
print(result.to_string(index=False))
