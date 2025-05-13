
# E-Commerce Customer Analysis using Olist Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# Load datasets
customers = pd.read_csv('data/olist_customers_dataset.csv')
orders = pd.read_csv('olist_orders_dataset.csv')
order_items = pd.read_csv('olist_order_items_dataset.csv')
payments = pd.read_csv('olist_order_payments_dataset.csv')

# Merge datasets
orders_customers = pd.merge(orders, customers, on='customer_id')
orders_customers_items = pd.merge(orders_customers, order_items, on='order_id')
full_data = pd.merge(orders_customers_items, payments, on='order_id')

# Convert date columns
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])

# Visualization: Customer count by state
plt.figure(figsize=(12,6))
state_counts = customers['customer_state'].value_counts().head(10)
sns.barplot(x=state_counts.index, y=state_counts.values)
plt.title('Top 10 Customer States')
plt.xlabel('State')
plt.ylabel('Number of Customers')
plt.show()

# Visualization: Order status distribution
plt.figure(figsize=(10,5))
order_status_counts = orders['order_status'].value_counts()
sns.barplot(x=order_status_counts.index, y=order_status_counts.values)
plt.title('Order Status Distribution')
plt.xlabel('Order Status')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Visualization: Payment method distribution
plt.figure(figsize=(8,4))
payment_types = payments['payment_type'].value_counts()
sns.barplot(x=payment_types.index, y=payment_types.values)
plt.title('Payment Method Distribution')
plt.xlabel('Payment Method')
plt.ylabel('Count')
plt.show()

# RFM Analysis
snapshot_date = orders['order_purchase_timestamp'].max() + dt.timedelta(days=1)
rfm_df = orders.groupby('customer_id').agg({
    'order_purchase_timestamp': lambda x: (snapshot_date - x.max()).days,
    'order_id': 'count'
}).reset_index()

rfm_df.columns = ['customer_id', 'Recency', 'Frequency']

# Merge with monetary value
monetary = order_items.groupby('order_id').agg({'price': 'sum'}).reset_index()
orders_monetary = pd.merge(orders[['order_id', 'customer_id']], monetary, on='order_id')
rfm_monetary = orders_monetary.groupby('customer_id').agg({'price': 'sum'}).reset_index()
rfm_monetary.columns = ['customer_id', 'Monetary']

# Final RFM Table
rfm = pd.merge(rfm_df, rfm_monetary, on='customer_id')

# Display top 10 RFM values
print("Top 10 Customers by RFM Metrics:")
print(rfm.head(10))

# Optional: Create RFM Segments (Simple Approach)
rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'], 4, labels=[1, 2, 3, 4])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4])
rfm['RFM_Segment'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

# Print Segment Counts
print("\nRFM Segment Distribution:")
print(rfm['RFM_Segment'].value_counts().head())
