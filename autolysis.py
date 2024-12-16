# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import missingno as msno
from scipy import stats
import requests
import os
import chardet
import sys

# --- ENVIRONMENT SETUP ---
if "AIPROXY_TOKEN" not in os.environ:
    api_key = input("Please enter your OpenAI API key: ")
    os.environ["AIPROXY_TOKEN"] = api_key
api_key = os.environ["AIPROXY_TOKEN"]

# --- UTILITY FUNCTIONS ---

def detect_encoding(filename):
    """
    Detect file encoding using chardet to handle CSV files with unknown encodings.
    
    Args:
        filename (str): Path to the CSV file.

    Returns:
        str: Detected encoding format.
    """
    with open(filename, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def load_and_clean_data(filename):
    """
    Load a dataset, handle missing values, and clean data dynamically based on data characteristics.

    Args:
        filename (str): Path to the CSV file.

    Returns:
        DataFrame: Cleaned dataset.
    """
    encoding = detect_encoding(filename)
    df = pd.read_csv(filename, encoding=encoding)

    # Dynamically handle missing data
    missing_data_percentage = df.isnull().mean() * 100
    if missing_data_percentage.max() > 40:  # Drop columns if more than 40% data is missing
        columns_to_drop = missing_data_percentage[missing_data_percentage > 40].index
        df.drop(columns=columns_to_drop, inplace=True)

    # Fill numeric and non-numeric missing values
    numeric_columns = df.select_dtypes(include='number')
    df[numeric_columns.columns] = numeric_columns.fillna(numeric_columns.mean())

    non_numeric_columns = df.select_dtypes(exclude='number')
    df[non_numeric_columns.columns] = non_numeric_columns.fillna('Unknown')

    # Drop completely empty rows
    df.dropna(axis=0, how='all', inplace=True)

    return df

def summarize_data(df):
    """
    Generate a summary of the dataset dynamically, adjusting based on the dataset structure.
    
    Args:
        df (DataFrame): The dataset.

    Returns:
        dict: Summary of the dataset.
    """
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'types': df.dtypes.to_dict(),
        'descriptive_statistics': df.describe().head().to_dict(),
        'missing_values': df.isnull().sum().to_dict()
    }
    return summary

def detect_outliers(df):
    """
    Identify outliers in numeric columns using Z-score analysis dynamically.
    
    Args:
        df (DataFrame): The dataset.

    Returns:
        dict: Number of outliers per numeric column (Z-score > 3).
    """
    numeric_df = df.select_dtypes(include=[np.number])
    z_scores = np.abs(stats.zscore(numeric_df))
    outliers = (z_scores > 3).sum(axis=0)
    outlier_info = {column: int(count) for column, count in zip(numeric_df.columns, outliers)}
    return outlier_info

def correlation_analysis(df):
    """
    Compute the correlation matrix for numeric columns dynamically, only computing top correlations.
    
    Args:
        df (DataFrame): The dataset.

    Returns:
        dict: Most significant correlations and p-values.
    """
    numeric_df = df.select_dtypes(include='number')
    correlation_matrix = numeric_df.corr()

    # Dynamically get top 5 most significant correlations
    correlation_matrix = correlation_matrix.abs().unstack().sort_values(ascending=False)
    significant_corr = correlation_matrix[(correlation_matrix < 1)].head(5)

    # Dynamically calculate statistical significance (p-value)
    p_values = numeric_df.corrwith(numeric_df).apply(lambda x: sm.OLS(x, sm.add_constant(np.arange(len(x)))).fit().pvalues[0])
    
    return significant_corr.to_dict(), p_values.head(5).to_dict()

def perform_clustering(df, n_clusters=None):
    """
    Perform KMeans clustering on numeric data dynamically based on the dataset.
    
    Args:
        df (DataFrame): The dataset.
        n_clusters (int, optional): Number of clusters to form. If None, will be determined dynamically.
    
    Returns:
        tuple: (Updated DataFrame with cluster labels, KMeans model)
    """
    # Determine optimal number of clusters dynamically (using the Elbow Method)
    if n_clusters is None:
        elbow_point = determine_optimal_clusters(df)
        n_clusters = elbow_point
    
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df_scaled)
    return df, kmeans

def determine_optimal_clusters(df):
    """
    Dynamically determine the optimal number of clusters using the Elbow Method.

    Args:
        df (DataFrame): The dataset.

    Returns:
        int: Optimal number of clusters.
    """
    from sklearn.cluster import KMeans
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    inertia = []
    
    for i in range(1, 11):  # Test for 1 to 10 clusters
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(df_scaled)
        inertia.append(kmeans.inertia_)
    
    # The optimal number of clusters is where the inertia starts decreasing less sharply
    elbow_point = np.argmin(np.diff(inertia)) + 2  # +2 as the index starts from 1
    return elbow_point

def perform_pca(df):
    """
    Perform Principal Component Analysis (PCA) dynamically based on dataset dimensions.
    
    Args:
        df (DataFrame): The dataset.
    
    Returns:
        DataFrame: Updated dataset with PCA components.
    """
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    pca = PCA(n_components=2)  # Fixed to 2 components for visualization
    pca_components = pca.fit_transform(df_scaled)
    df['PCA1'] = pca_components[:, 0]
    df['PCA2'] = pca_components[:, 1]
    return df

def create_visualizations(df):
    """
    Generate and save visualizations dynamically, adjusting for available data.
    
    Args:
        df (DataFrame): The dataset.
    
    Returns:
        list: List of file paths for generated images.
    """
    msno.matrix(df)
    missing_img = 'missing_data.png'
    plt.tight_layout()
    plt.savefig(missing_img)
    plt.close()

    numeric_df = df.select_dtypes(include='number')
    correlation_img = None
    if numeric_df.shape[1] > 1:
        plt.figure(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title("Correlation Matrix")
        correlation_img = 'correlation_matrix.png'
        plt.tight_layout()
        plt.savefig(correlation_img)
        plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='Set1', s=100, edgecolor='black')
    plt.title("Cluster Analysis (PCA)")
    cluster_img = 'cluster_analysis.png'
    plt.tight_layout()
    plt.savefig(cluster_img)
    plt.close()

    return [missing_img, correlation_img, cluster_img] if correlation_img else [missing_img, cluster_img]

def get_ai_story(dataset_summary, significant_correlations, visualizations, dynamic_info, iteration=1):
    """
    Generate a dynamic narrative analysis using the OpenAI API based on dataset findings and dynamic elements.
    
    Args:
        dataset_summary (dict): Summary of the dataset.
        significant_correlations (dict): Most significant correlations.
        visualizations (list): List of generated visualization paths.
        dynamic_info (dict): Dynamically generated information (e.g., optimal clusters, outliers).
        iteration (int): The iteration count (default is 1).

    Returns:
        str: Generated narrative from the LLM.
    """
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {os.environ['AIPROXY_TOKEN']}"}

    prompt = f"""
    Iteration {iteration}: Generate a dynamic narrative based on the following dataset analysis:
    **Dataset Summary**: {dataset_summary}
    **Key Correlations (Top 5)**: {significant_correlations}
    **Dynamic Insights**: {dynamic_info}
    **Visualizations**: {visualizations}
    Please emphasize key findings, such as correlations, cluster analysis, outlier handling, and missing data strategies.
    """

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "system", "content": "You are a helpful assistant."}, 
                     {"role": "user", "content": prompt}]
    }

    response = requests.post(url, json=payload, headers=headers)
    return response.json()['choices'][0]['message']['content']

def main(dataset_name):
    """
    Main function to load the dataset, clean, analyze, visualize, and generate insights dynamically.

    Args:
        dataset_name (str): Dataset file name (without extension).
    """
    file_path = f"{dataset_name}.csv"
    df = load_and_clean_data(file_path)
    
    dataset_summary = summarize_data(df)
    significant_correlations, p_values = correlation_analysis(df)
    outlier_info = detect_outliers(df)
    df_with_clusters, kmeans = perform_clustering(df)
    df_with_pca = perform_pca(df_with_clusters)

    visualizations = create_visualizations(df_with_pca)
    dynamic_info = {
        'outliers': outlier_info,
        'optimal_clusters': kmeans.n_clusters,
        'significant_p_values': p_values
    }

    narrative = get_ai_story(dataset_summary, significant_correlations, visualizations, dynamic_info)
    print(narrative)

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    main(dataset_name)
