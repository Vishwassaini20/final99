# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "chardet",
#     "matplotlib",
#     "pandas",
#     "statsmodels",
#     "scikit-learn",
#     "missingno",
#     "python-dotenv",
#     "requests",
#     "seaborn",
# ]
# ///


## This script does generic analysis which includes summarization, cluster analysis, Correlation analysis,
## outlier analysis along with Visulization using python of any csv files and results of this analysis is shared with LLM Model
## to come up with a story and the results are stored a README.md file

# -*- coding: utf-8 -*-
"""autolysis.ipynb"""

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

# Set the AIPROXY_TOKEN environment variable if not already set
if "AIPROXY_TOKEN" not in os.environ:
    api_key = input("Please enter your OpenAI API key: ")
    os.environ["AIPROXY_TOKEN"] = api_key

api_key = os.environ["AIPROXY_TOKEN"]

# Function to detect file encoding
def detect_encoding(filename):
    with open(filename, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

# Function to load and clean the dataset
def load_and_clean_data(filename):
    encoding = detect_encoding(filename)
    df = pd.read_csv(filename, encoding=encoding)
    
    df.dropna(axis=0, how='all', inplace=True)
    numeric_columns = df.select_dtypes(include='number')
    df[numeric_columns.columns] = numeric_columns.fillna(numeric_columns.mean())
    non_numeric_columns = df.select_dtypes(exclude='number')
    df[non_numeric_columns.columns] = non_numeric_columns.fillna('Unknown')
    return df

# Function to summarize the dataset
def summarize_data(df):
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'types': df.dtypes.to_dict(),
        'descriptive_statistics': df.describe().to_dict(),
        'missing_values': df.isnull().sum().to_dict()
    }
    return summary

# Outlier detection function using Z-Score
def detect_outliers(df):
    numeric_df = df.select_dtypes(include=[np.number])
    z_scores = np.abs(stats.zscore(numeric_df))
    outliers = (z_scores > 3).sum(axis=0)
    outlier_info = {column: int(count) for column, count in zip(numeric_df.columns, outliers)}
    return outlier_info

# Correlation analysis function
def correlation_analysis(df):
    numeric_df = df.select_dtypes(include='number')
    correlation_matrix = numeric_df.corr()
    return correlation_matrix.to_dict()

# Cluster analysis using KMeans
def perform_clustering(df, n_clusters=3):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df_scaled)
    return df, kmeans

# PCA for dimensionality reduction (optional)
def perform_pca(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(df_scaled)
    df['PCA1'] = pca_components[:, 0]
    df['PCA2'] = pca_components[:, 1]
    return df

# Function to create visualizations
def create_visualizations(df):
    msno.matrix(df)
    plt.tight_layout()
    missing_img = 'missing_data.png'
    plt.savefig(missing_img)
    plt.close()

    numeric_df = df.select_dtypes(include='number')
    if numeric_df.shape[1] > 1:
        plt.figure(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Matrix")
        correlation_img = 'correlation_matrix.png'
        plt.tight_layout()
        plt.savefig(correlation_img)
        plt.close()
    else:
        correlation_img = None

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='Set1')
    plt.title("Cluster Analysis (PCA)")
    cluster_img = 'cluster_analysis.png'
    plt.tight_layout()
    plt.savefig(cluster_img)
    plt.close()

    return [missing_img, correlation_img, cluster_img] if correlation_img else [missing_img, cluster_img]

# Replacing generate_analysis_story with get_ai_story
def get_ai_story(dataset_summary, dataset_info, visualizations):
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {os.environ['AIPROXY_TOKEN']}"}

    prompt = f"""
    Below is a detailed summary and analysis of a dataset. Please generate a **rich and engaging narrative** about this dataset analysis, including:

    1. **The Data Received**: Describe the dataset vividly. What does the data represent? What are its features? What is the significance of this data? Create a compelling story around it.
    2. **The Analysis Carried Out**: Explain the analysis methods used. Highlight techniques like missing value handling, outlier detection, clustering, and dimensionality reduction (PCA). How do these methods provide insights?
    3. **Key Insights and Discoveries**: What were the major findings? What trends or patterns emerged that can be interpreted as discoveries? Were there any unexpected results?
    4. **Implications and Actions**: Discuss the implications of these findings. How do they influence decisions? What actionable recommendations would you provide based on the analysis?
    5. **Visualizations**: Describe the visualizations included. What do they reveal about the data? How do they complement the analysis and findings?

    **Dataset Summary**:
    {dataset_summary}

    **Dataset Info**:
    {dataset_info}

    **Visualizations**:
    {visualizations}
    """

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        story = result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"Error generating analysis story: {e}")
        return "Error: Unable to generate narrative. Please check the AI service."

    return story

# Function to write the README
def write_readme(summary, outliers, correlation_matrix, visualizations, story, filename):
    with open('README.md', 'w') as f:
        f.write(f"# Dataset Analysis of {filename}\n")
        f.write("\n## Dataset Summary\n")
        f.write(f"- Shape of the dataset: {summary['shape']}\n")
        f.write(f"- Columns: {', '.join(summary['columns'])}\n")
        f.write(f"- Data types:\n{summary['types']}\n")
        f.write(f"- Descriptive statistics:\n{summary['descriptive_statistics']}\n")
        f.write(f"- Missing values per column:\n{summary['missing_values']}\n")
        f.write("\n## Outlier Detection\n")
        f.write(f"Outliers detected in each numeric column (Z-score > 3):\n{outliers}\n")
        f.write("\n## Correlation Analysis\n")
        f.write(f"Correlation Matrix:\n{correlation_matrix}\n")
        f.write("\n## Dataset Analysis Story\n")
        f.write(f"{story}\n")
        f.write("\n## Visualizations\n")
        for img in visualizations:
            f.write(f"![{img}]({img})\n")

# Main function
def main(filename):
    df = load_and_clean_data(filename)
    summary = summarize_data(df)
    outliers = detect_outliers(df)
    correlation_matrix = correlation_analysis(df)
    df, kmeans = perform_clustering(df)
    df = perform_pca(df)
    visualizations = create_visualizations(df)
    story = get_ai_story(summary, df.head().to_dict(), visualizations)
    write_readme(summary, outliers, correlation_matrix, visualizations, story, filename)
    print(f"Analysis complete. Results saved in 'README.md'.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
    else:
        main(sys.argv[1])
