"""
data_eda.py - EDA   functions for credit risk probability modeling

This module contains functions to eda performing functions for credit risk probability modeling

Author: [Metasebiya Bizuneh]
Created: June 28, 2025
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from data_loader import DataLoader

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from data_loader import DataLoader

class Dataprocessor:
    def __init__(self, data,
                 output_path: str = "../data/processed/filtered_complaints.csv",
                 target_products: list = None):
        self.df = data
        self.output_path = output_path
        self.target_products = target_products or [
            "Credit card", "Personal loan", "Buy Now, Pay Later (BNPL)",
            "Savings account", "Money transfers"
        ]
        self.product_map = product_map = {
            "credit card": "Credit card",
            "credit card or prepaid card": "Credit card",
            "payday loan, title loan, or personal loan": "Personal loan",
            "payday loan, title loan, personal loan, or advance loan": "Personal loan",
            "consumer loan": "Personal loan",
            "money transfers": "Money transfers",
            "money transfer, virtual currency, or money service": "Money transfers",
            "checking or savings account": "Savings account",
            "bank account or service": "Savings account",
            "buy now, pay later (bnpl)": "Buy Now, Pay Later (BNPL)"  # if found
    }

    def standardize_columns(self):
        self.df.columns = (
            self.df.columns
            .str.lower()
            .str.replace(r'[\s\-]+', '_', regex=True)
            .str.strip('_')
        )

    def convert_types(self):
        if 'date_received' in self.df.columns:
            self.df['date_received'] = pd.to_datetime(self.df['date_received'], errors='coerce')

    def overview_data(self, title="Dataset Overview") -> pd.DataFrame:
        print(f"\n📊 {title}")
        print(self.df.describe(include='all'))
        print("\n📋 Columns:", self.df.columns.tolist())
        print(f"\n🧱 Info: {self.df.info()}")
        print(f"🧱 Shape: {self.df.shape}")
        print(f"📦 Total Elements: {self.df.size}")
        print("\n🧾 Missing Values Per Column:")
        print(self.df.isnull().sum())
        print(f"\n🔍 Duplicate Rows: {self.df.duplicated().sum()}")
        print(f"\n🔍 product size: {self.df.groupby('product').size()}")
        return self.df

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def clean_missing_and_duplicates(self, essential_columns=None):
        print("\n🧹 Cleaning duplicates and missing values...")
        if essential_columns is None:
            essential_columns = ['consumer_complaint_narrative']

        initial_shape = self.df.shape
        print(f"✅  {initial_shape[0]} rows.")
        # self.df = self.df.drop_duplicates()
        # print(f"✅ Removed {initial_shape[0] - self.df.shape[0]} rows.")
        self.df = self.df.dropna(subset=essential_columns)
        print(f"✅ Removed {initial_shape[0] - self.df.shape[0]} rows.")
        # Normalize and map to target product categories
        self.df['product'] = self.df['product'].str.strip().str.lower().map(self.product_map)
        print(f"✅ Removed {initial_shape[0] - self.df.shape[0]} rows.")

    def plot_distributions(self):
        if 'product' in self.df.columns:
            plt.figure(figsize=(8, 4))
            sns.countplot(data=self.df, x='product', order=self.df['product'].value_counts().index)
            plt.title('Product Distribution')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            plt.figure(figsize=(6, 4))
            sns.histplot(self.df[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.tight_layout()
            plt.show()

    def correlation_heatmap(self):
        corr = self.df.select_dtypes(include=['float64', 'int64']).corr()
        if not corr.empty:
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
            plt.title("Correlation Matrix")
            plt.tight_layout()
            plt.show()

    def filter_and_clean_complaints(self) -> pd.DataFrame:
        print("🔍 Filtering by products and non-empty narratives...")

        self.clean_missing_and_duplicates()

        # Drop rows that don't match mapped categories
        self.df = self.df[self.df['product'].notnull()]
        print(f"🧱 Shape: {self.df.shape}")

        df_filtered = self.df.copy()

        print(f"✅ Filtered to {df_filtered.shape[0]} rows.")
        print("🧹 Cleaning narratives...")
        df_filtered['cleaned_narrative'] = df_filtered['consumer_complaint_narrative'].apply(self.clean_text)

        cols_to_keep = ['complaint_id', 'product', 'company', 'date_received', 'cleaned_narrative']
        df_filtered = df_filtered[[col for col in cols_to_keep if col in df_filtered.columns]]

        print("💾 Saving to CSV...")
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df_filtered.to_csv(self.output_path, index=False)
        print(f"🎉 Cleaned data saved to {self.output_path}")

        self.df = df_filtered  # update internal state
        return df_filtered

    def detect_bnpl_from_narrative(self):
        """
        Detects BNPL-related complaints based on narrative text.
        Adds a column `is_bnpl` and updates product label if matched.
        """
        print("🔍 Searching for BNPL-related complaints in narratives...")

        bnpl_keywords = [
            "affirm", "klarna", "afterpay", "zip", "sezzle",
            "buy now pay later", "bnpl", "split payment", "pay in 4", "pay later"
        ]
        # Combine keywords into one regex pattern
        pattern = '|'.join([re.escape(kw) for kw in bnpl_keywords])

        # Lowercase narrative for search
        self.df['narrative_lower'] = self.df['consumer_complaint_narrative'].str.lower()

        # Mark as BNPL if any keyword matches
        self.df['is_bnpl'] = self.df['narrative_lower'].str.contains(pattern, na=False)

        # Update product label
        matched_rows = self.df['is_bnpl'].sum()
        print(f"✅ Detected {matched_rows} BNPL-related complaints.")

        self.df.loc[self.df['is_bnpl'], 'product'] = 'Buy Now, Pay Later (BNPL)'
        self.df.drop(columns=['narrative_lower'], inplace=True)
        print(self.df[self.df['is_bnpl']]['consumer_complaint_narrative'].sample(5).tolist())


if __name__ == "__main__":
    df_path = "../data/raw/complaints.csv"
    data = DataLoader().load_data(df_path)
    processor = Dataprocessor(data)
    processor.standardize_columns()
    processor.convert_types()
    processor.detect_bnpl_from_narrative()
    processor.overview_data("Raw Complaint Dataset Overview")
    processor.filter_and_clean_complaints()
    processor.overview_data("Filtered Dataset Overview")