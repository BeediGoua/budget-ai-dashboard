import pandas as pd
import numpy as np
import re


# === 1. EXTRACTION DES FEATURES DE DATE ===
def extract_date_features(df):
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Day'] = df['Date'].dt.day
    df['Hour'] = df['Date'].dt.hour
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
    return df


# === 2. ENCODAGE CYCLIQUE (jour, heure) ===
def encode_day_of_week_cyclic(df):
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    return df

def encode_hour_cyclic(df):
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    return df


# === 3. NETTOYAGE DE DESCRIPTION POUR NLP ===
def clean_description(df):
    df['Cleaned_Description'] = df['Description'].astype(str).str.lower().str.strip()
    df['Cleaned_Description'] = df['Cleaned_Description'].apply(lambda x: re.sub(r'[^a-z0-9\s]', '', x))
    df['Cleaned_Description'] = df['Cleaned_Description'].replace('', np.nan)
    return df


# === 4. FLAG DE TRANSACTION IMPORTANTE (anomalies) ===
def create_transaction_flags(df):
    threshold = df["Amount"].quantile(0.95)
    df["IsLargeAmount"] = (df["Amount"] > threshold).astype(int)
    df["IsVeryLargeAmount"] = (df["Amount"] > df["Amount"].quantile(0.99)).astype(int)
    return df


# === 5. CATEGORISATION SIMPLIFIEE INITIALE ===
def categorize_amount_level(df):
    df['AmountLevel'] = pd.qcut(df['Amount'], q=4, labels=['small', 'medium', 'large', 'very_large'])
    return df


# === 6. CALCUL DU TAUX D’ÉPARGNE MENSUEL ===
def calculate_saving_rate(df):
    # Détection du nom correct de la colonne
    transaction_col = 'Transaction Type' if 'Transaction Type' in df.columns else 'Type'
    
    df[transaction_col] = df[transaction_col].str.lower()
    monthly = df.groupby(['Year', 'Month', transaction_col])['Amount'].sum().unstack().fillna(0)
    monthly['SavingRate'] = (monthly.get('credit', 0) - monthly.get('debit', 0)) / (monthly.get('credit', 0) + 1e-6)
    return monthly.reset_index()


# === 7. TYPE DE TRANSACTION : FIXE VS VARIABLE (à estimer par fréquence) ===
def tag_fixed_expenses(df, min_occurrence=5):
    """Ajoute une colonne IsRecurring basée sur les descriptions les plus fréquentes"""
    desc_counts = df['Cleaned_Description'].value_counts()
    recurring = desc_counts[desc_counts >= min_occurrence].index
    df['IsRecurring'] = df['Cleaned_Description'].isin(recurring).astype(int)
    return df
