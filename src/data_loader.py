"""
data_loader.py - Data loading functions for intelligent complaint analysis

This module contains functions to load data for intelligent complaint analysis

Author: [Metasebiya Bizuneh]
Created: July 3, 2025
"""

import os
import pandas as pd


class DataLoader:
    # def __init__(self, customer_review):
    #     self.customer_review = customer_review

    def load_data(self, file_path):
        """
        Load a CSV file into a pandas DataFrame for financial analysis

        Parameters:
            file_path (str): Path to the CSV file (e.g., 'data/raw/{app_name}.csv')

        Returns:
            pd.DataFrame: Loaded customer review data as a DataFrame

        Raises:
            FileNotFoundError: If the specified file does not exist
        """

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist")
        df = pd.read_csv(file_path)
        return df

    def load_cleaned_data(self, file_path):
        """
        Load a CSV file into a pandas DataFrame for financial analysis

        Parameters:
            file_path (str): Path to the CSV file (e.g., 'data/raw/{app_name}.csv')

        Returns:
            pd.DataFrame: Loaded customer review data as a DataFrame

        Raises:
            FileNotFoundError: If the specified file does not exist
        """

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist")
        df = pd.read_csv(file_path)
        print(df.head())
        return df


if __name__ == "__main__":
    df = "../data/raw/complaints.csv"
    data = DataLoader()
    all_data = data.load_data(df)
    print(all_data)