#!/usr/bin/env python3
"""
Anthropometric Data Analyzer with Tactical Helmet Design
Integrates general anthropometric analysis with MIL-STD-1472 compliant helmet design

This is the main entry point for the application, integrating:
1. General anthropometric data analysis
2. MIL-STD-1472 compliant tactical helmet design
3. Individual helmet fit analysis 
"""

import os
import numpy as np
import pandas as pd
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, send_file, jsonify, request, redirect, url_for
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging
import json
from werkzeug.utils import secure_filename
from matplotlib.lines import Line2D
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Ensure output directories exist
os.makedirs('helmet_design_outputs', exist_ok=True)
os.makedirs('static/js', exist_ok=True)
os.makedirs('static/css', exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "anthropometric-app-secret")

# Define styling constants for visualization
MILITARY_GREEN = '#4A6741'
MILITARY_YELLOW = '#D4B94E'
MILITARY_RED = '#8F4539'

# ----- GENERAL ANTHROPOMETRIC DATA FUNCTIONS -----

def calculate_correlation_analysis(df):
    """
    Perform correlation analysis on anthropometric data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing anthropometric measurements
        
    Returns:
    --------
    correlation_matrix : pandas.DataFrame
        Matrix of Pearson correlation coefficients
    correlation_plot : str
        Base64-encoded correlation heatmap image
    key_insights : list
        List of key correlation insights
    """
    # Drop non-numeric columns if present
    if 'Gender' in df.columns:
        df_numeric = df.drop('Gender', axis=1)
    else:
        df_numeric = df
    
    # Calculate correlation matrix
    correlation_matrix = df_numeric.corr().round(3)
    
    # Generate correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    cmap = plt.cm.RdBu_r
    
    # Draw heatmap
    heatmap = plt.pcolormesh(correlation_matrix, cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(heatmap, label='Correlation Coefficient')
    plt.title('Correlation Heatmap of Anthropometric Measurements', fontsize=14)
    
    # Set ticks and labels
    plt.xticks(np.arange(0.5, len(correlation_matrix.columns), 1), correlation_matrix.columns, rotation=90)
    plt.yticks(np.arange(0.5, len(correlation_matrix.columns), 1), correlation_matrix.columns)
    
    # Save as base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=80)
    buffer.seek(0)
    correlation_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    # Extract key insights
    key_insights = []
    
    # Find strongest positive correlations
    pos_corr = correlation_matrix.unstack().sort_values(ascending=False)
    pos_corr = pos_corr[pos_corr < 1.0]  # Remove self-correlations (value = 1.0)
    
    # Find strongest negative correlations
    neg_corr = correlation_matrix.unstack().sort_values(ascending=True)
    
    # Add top 5 positive correlations as insights
    key_insights.append("Strongest positive correlations:")
    for idx, val in pos_corr[:5].items():
        key_insights.append(f"{idx[0]} & {idx[1]}: {val:.3f}")
    
    # Add top 5 negative correlations as insights (if any negative exists)
    if neg_corr.min() < 0:
        key_insights.append("\nStrongest negative correlations:")
        for idx, val in neg_corr[:5].items():
            if val < 0:  # Only include negative correlations
                key_insights.append(f"{idx[0]} & {idx[1]}: {val:.3f}")
    
    return correlation_matrix, correlation_plot, key_insights

def perform_frequency_analysis(df, columns=None):
    """
    Perform frequency distribution analysis on selected measurements
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing anthropometric measurements
    columns : list, optional
        List of column names to analyze, if None, select key measurements
        
    Returns:
    --------
    histograms : dict
        Dictionary of base64-encoded histogram images for each measurement
    summary : dict
        Dictionary containing distribution statistics for each measurement
    """
    if columns is None:
        columns = ['Head circumference', 'Head breadth', 'Head length', 'Total head height']
    
    histograms = {}
    summary = {}
    
    for column in columns:
        if column in df.columns:
            # Create histogram with density curve
            plt.figure(figsize=(8, 6))
            
            # Plot histogram and get the bin values
            n, bins, patches = plt.hist(df[column], bins=15, density=True, alpha=0.7, color=MILITARY_GREEN)
            
            # Add density curve
            from scipy import stats
            kde = stats.gaussian_kde(df[column])
            x = np.linspace(df[column].min(), df[column].max(), 200)
            plt.plot(x, kde(x), 'r-', linewidth=2)
            
            # Add vertical lines for percentiles
            percentiles = [5, 50, 95]
            percentile_values = np.percentile(df[column], percentiles)
            line_styles = ['--', '-', '--']
            colors = [MILITARY_RED, 'black', MILITARY_RED]
            labels = ['5th percentile', 'Median', '95th percentile']
            
            for p, pv, ls, c, lbl in zip(percentiles, percentile_values, line_styles, colors, labels):
                plt.axvline(x=pv, linestyle=ls, color=c, label=f"{lbl}: {pv:.1f} mm")
            
            plt.title(f'Distribution of {column}')
            plt.xlabel('Measurement (mm)')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Save as base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=80)
            buffer.seek(0)
            histograms[column] = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            # Calculate distribution statistics
            summary[column] = {
                'mean': df[column].mean().round(2),
                'median': df[column].median().round(2),
                'std': df[column].std().round(2),
                'min': df[column].min().round(2),
                'max': df[column].max().round(2),
                'p5': np.percentile(df[column], 5).round(2),
                'p95': np.percentile(df[column], 95).round(2),
                'range': (df[column].max() - df[column].min()).round(2),
                'iqr': (df[column].quantile(0.75) - df[column].quantile(0.25)).round(2),
                'skewness': stats.skew(df[column]).round(3),
                'kurtosis': stats.kurtosis(df[column]).round(3)
            }
    
    return histograms, summary

def create_dataframe_male():
    """Generate synthetic anthropometric measurement data for males"""
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.precision", 3)
    np.set_printoptions(precision=3, suppress=True)

    np.random.seed(0)
    columns = [
        "Total head height", "Stomion to vertex height", "Pronasale to vertex height", "Inion to vertex height",
        "Right tragion to vertex height", "Right ectocanthion to vertex height", "Face length", "Glabella to vertex height",
        "Menton to subnasale length", "Nose length", "Head circumference", "Head length",
        "Right tragion to ectocanthion length", "Head breadth", "Frontotemporale breadth", "Bitragion breadth",
        "Bizygomatic breadth", "Biocular breadth", "Interpupillary distance", "Interocular breadth",
        "Nasal root breadth", "Nose breadth", "Mouth breadth", "Bigonial breadth",
        "Right tragion to ectocanthion breadth", "Ear length", "Sagittal arc", "Bitragion arc",
        "Morphologic face height", "Left ectocanthion to vertex height", "Left tragion to vertex height",
        "Occiput to left ectocanthion distance", "Occiput to left tragion distance", "Occiput to stomion distance",
        "Glabella to subnasale height", "Physiognomic face height", "Occiput to right tragion distance",
        "Occiput to right ectocanthion distance", "Sellion to vertex height", "Subnasale to vertex height"
    ]
    male_means = np.array([
        230, 160, 170, 180, 175, 165, 200, 155, 70, 50, 570, 210, 45, 150, 140, 160, 135, 65, 60, 40,
        30, 35, 80, 120, 50, 60, 220, 230, 180, 150, 150, 180, 185, 190, 120, 190, 175, 175, 140, 160
    ])
    male_stds = np.array([
        10, 8, 8, 8, 7, 7, 10, 7, 5, 4, 15, 10, 4, 6, 6, 6, 5, 4, 3, 3,
        2, 2, 4, 6, 3, 4, 12, 12, 10, 5, 5, 7, 7, 8, 4, 8, 7, 7, 5, 5
    ])
    num_samples = 100  # Sample size for males: 100
    data = np.random.normal(loc=male_means, scale=male_stds, size=(num_samples, len(columns)))
    return pd.DataFrame(data, columns=columns)

def create_dataframe_female():
    """Generate synthetic anthropometric measurement data for females"""
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.precision", 3)
    np.set_printoptions(precision=3, suppress=True)
    
    np.random.seed(1)
    columns = [
        "Total head height", "Stomion to vertex height", "Pronasale to vertex height", "Inion to vertex height",
        "Right tragion to vertex height", "Right ectocanthion to vertex height", "Face length", "Glabella to vertex height",
        "Menton to subnasale length", "Nose length", "Head circumference", "Head length",
        "Right tragion to ectocanthion length", "Head breadth", "Frontotemporale breadth", "Bitragion breadth",
        "Bizygomatic breadth", "Biocular breadth", "Interpupillary distance", "Interocular breadth",
        "Nasal root breadth", "Nose breadth", "Mouth breadth", "Bigonial breadth",
        "Right tragion to ectocanthion breadth", "Ear length", "Sagittal arc", "Bitragion arc",
        "Morphologic face height", "Left ectocanthion to vertex height", "Left tragion to vertex height",
        "Occiput to left ectocanthion distance", "Occiput to left tragion distance", "Occiput to stomion distance",
        "Glabella to subnasale height", "Physiognomic face height", "Occiput to right tragion distance",
        "Occiput to right ectocanthion distance", "Sellion to vertex height", "Subnasale to vertex height"
    ]
    male_means = np.array([
        230, 160, 170, 180, 175, 165, 200, 155, 70, 50, 570, 210, 45, 150, 140, 160, 135, 65, 60, 40,
        30, 35, 80, 120, 50, 60, 220, 230, 180, 150, 150, 180, 185, 190, 120, 190, 175, 175, 140, 160
    ])
    male_stds = np.array([
        10, 8, 8, 8, 7, 7, 10, 7, 5, 4, 15, 10, 4, 6, 6, 6, 5, 4, 3, 3,
        2, 2, 4, 6, 3, 4, 12, 12, 10, 5, 5, 7, 7, 8, 4, 8, 7, 7, 5, 5
    ])
    modifications = np.array([
        56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
        56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56
    ])
    female_means = np.array([
        218.4, 182.3, 176.5, 169.8, 164.2, 158.7, 187.4, 166.9, 67.2, 52.8, 558.3, 186.5, 42.3, 146.8, 134.2, 
        154.6, 132.4, 63.5, 58.2, 37.8, 30.4, 34.6, 77.3, 114.5, 47.2, 57.8, 212.4, 223.6, 169.3, 158.7, 
        164.2, 172.5, 177.8, 183.2, 115.6, 182.4, 177.8, 172.5, 162.3, 168.7
    ])
    female_stds = male_stds * 0.9
    num_samples = 56  # Sample size for females: 56
    data = np.random.normal(loc=female_means, scale=female_stds, size=(num_samples, len(columns)))
    return pd.DataFrame(data, columns=columns)

def create_dataframe(gender='both'):
    """Create dataframe based on gender selection"""
    if gender == 'male':
        return create_dataframe_male()
    elif gender == 'female':
        return create_dataframe_female()
    else:
        # Combine male and female data
        male_df = create_dataframe_male()
        female_df = create_dataframe_female()
        
        # Add gender column
        male_df['Gender'] = 'Male'
        female_df['Gender'] = 'Female'
        
        # Combine data
        combined_df = pd.concat([male_df, female_df], ignore_index=True)
        return combined_df

def perform_cluster_analysis(df, n_clusters=3, population_coverage=0.9):
    """
    Perform cluster analysis on anthropometric data following MIL-STD-1472 requirements
    to ensure coverage of at least 90% of the population
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing anthropometric measurements
    n_clusters : int, optional
        Number of clusters to identify
    population_coverage : float, optional
        Minimum population coverage required (default 0.9 for 90%)
        
    Returns:
    --------
    cluster_df : pandas.DataFrame
        Original dataframe with cluster assignments
    cluster_plot : str
        Base64-encoded cluster visualization plot
    cluster_stats : dict
        Dictionary of cluster statistics and MIL-STD-1472 compliance
    """
    from sklearn.cluster import KMeans
    from scipy import stats
    
    # Drop non-numeric columns if present
    if 'Gender' in df.columns:
        df_numeric = df.drop('Gender', axis=1)
    else:
        df_numeric = df
    
    # Key measurements according to MIL-STD-1472
    mil_std_key_measurements = [
        'Head circumference', 'Head breadth', 'Head length', 
        'Total head height', 'Bitragion breadth', 'Interpupillary breadth'
    ]
    
    # Filter for key measurements that exist in our dataset
    key_measurements = [m for m in mil_std_key_measurements if m in df_numeric.columns]
    
    # If no key measurements are found, use all available measurements
    if not key_measurements:
        key_measurements = df_numeric.columns.tolist()
    
    # Use only key measurements for clustering to focus on MIL-STD-1472 requirements
    clustering_data = df_numeric[key_measurements]
    
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(clustering_data)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(data_scaled)
    
    # Add cluster labels to original dataframe
    cluster_df = df.copy()
    cluster_df['Cluster'] = cluster_labels
    
    # Calculate cluster statistics
    cluster_stats = {}
    cluster_percentile_coverage = {}
    
    for cluster in range(n_clusters):
        cluster_data = cluster_df[cluster_df['Cluster'] == cluster]
        
        # Calculate key statistics for this cluster
        cluster_stats[f'Cluster {cluster}'] = {
            'count': len(cluster_data),
            'percentage': f"{(len(cluster_data) / len(cluster_df) * 100):.1f}%",
            'key_measurements': {},
            'mil_std_compliance': {}
        }
        
        # For each key measurement, calculate mean values and percentile range
        for measure in key_measurements:
            if measure in cluster_data.columns:
                # Basic statistics
                mean_val = cluster_data[measure].mean().round(2)
                std_val = cluster_data[measure].std().round(2)
                
                # Calculate 5th and 95th percentiles for MIL-STD-1472 compliance
                p5 = np.percentile(cluster_data[measure], 5).round(2)
                p95 = np.percentile(cluster_data[measure], 95).round(2)
                
                # Store statistics
                cluster_stats[f'Cluster {cluster}']['key_measurements'][measure] = {
                    'mean': mean_val,
                    'std': std_val,
                    'p5': p5,
                    'p95': p95,
                    'range': f"{p5} - {p95} mm"
                }
                
                # Check if this cluster covers enough of the population for this measurement
                # Calculate what percentage of the overall population falls within this cluster's range
                overall_pop_in_range = ((df_numeric[measure] >= p5) & (df_numeric[measure] <= p95)).mean()
                
                # Store compliance information
                cluster_stats[f'Cluster {cluster}']['mil_std_compliance'][measure] = {
                    'population_coverage': f"{overall_pop_in_range:.2%}",
                    'meets_90_percent': overall_pop_in_range >= population_coverage
                }
                
                # Track the coverage for overall assessment
                if measure not in cluster_percentile_coverage:
                    cluster_percentile_coverage[measure] = []
                cluster_percentile_coverage[measure].append(overall_pop_in_range)
    
    # Calculate overall MIL-STD-1472 compliance
    mil_std_compliance = {
        'overall_assessment': "Based on MIL-STD-1472 requirements for 90% population coverage:",
        'measurements': {}
    }
    
    for measure, coverages in cluster_percentile_coverage.items():
        max_coverage = max(coverages)
        compliant = max_coverage >= population_coverage
        
        mil_std_compliance['measurements'][measure] = {
            'max_coverage': f"{max_coverage:.2%}",
            'compliant': compliant,
            'assessment': f"{'✓ Compliant' if compliant else '✗ Non-compliant'} with MIL-STD-1472 (90% requirement)"
        }
    
    # Add overall compliance to cluster stats
    cluster_stats['mil_std_compliance'] = mil_std_compliance
    
    # Visualize clusters using PCA for dimensionality reduction
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_scaled)
    
    # Create a scatter plot of the clusters
    plt.figure(figsize=(12, 10))
    colors = ['#4A6741', '#D4B94E', '#8F4539', '#6082B6', '#AB92BF']
    
    # Draw ellipses to show 90% coverage for each cluster
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    
    for cluster in range(n_clusters):
        # Get points in this cluster
        cluster_points = pca_result[cluster_labels == cluster]
        
        # Plot the points
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   alpha=0.7, s=50, 
                   color=colors[cluster % len(colors)],
                   label=f'Cluster {cluster}')
        
        # Calculate mean and covariance
        if len(cluster_points) > 1:  # Need at least 2 points for covariance
            cluster_mean = np.mean(cluster_points, axis=0)
            cluster_cov = np.cov(cluster_points, rowvar=False)
            
            # Calculate eigenvalues and eigenvectors
            eigenvals, eigenvecs = np.linalg.eigh(cluster_cov)
            
            # Sort eigenvectors by eigenvalues in descending order
            order = eigenvals.argsort()[::-1]
            eigenvals = eigenvals[order]
            eigenvecs = eigenvecs[:, order]
            
            # Calculate angle and scale for ellipse
            theta = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))
            width, height = 2 * np.sqrt(eigenvals) * 2.4477  # 2.4477 for 90% coverage
            
            # Create ellipse
            ellipse = Ellipse(xy=cluster_mean, width=width, height=height,
                             angle=theta, edgecolor=colors[cluster % len(colors)],
                             facecolor='none', linestyle='--', linewidth=2, alpha=0.7)
            plt.gca().add_patch(ellipse)
    
    plt.title('MIL-STD-1472 Compliant Clustering Analysis\n(90% Population Coverage)', fontsize=14)
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotation about MIL-STD-1472
    compliant_measures = sum(1 for m in mil_std_compliance['measurements'].values() if m['compliant'])
    total_measures = len(mil_std_compliance['measurements'])
    
    compliance_text = f"MIL-STD-1472 Compliance: {compliant_measures}/{total_measures} measurements\n"
    compliance_text += f"Dashed lines represent 90% coverage boundaries"
    
    plt.annotate(compliance_text, xy=(0.5, 0.01), xycoords='figure fraction',
                horizontalalignment='center', verticalalignment='bottom',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
    
    # Save as base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=80)
    buffer.seek(0)
    cluster_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return cluster_df, cluster_plot, cluster_stats

def detect_outliers(df, method='zscore', threshold=3.0):
    """
    Detect outliers in anthropometric data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing anthropometric measurements
    method : str, optional
        Method to use for outlier detection: 'zscore' or 'iqr'
    threshold : float, optional
        Threshold for Z-score method
        
    Returns:
    --------
    outliers_df : pandas.DataFrame
        DataFrame containing outlier information
    outlier_plot : str
        Base64-encoded outlier visualization plot
    summary : dict
        Summary of outlier analysis
    """
    # Drop non-numeric columns if present
    if 'Gender' in df.columns:
        df_numeric = df.drop('Gender', axis=1)
    else:
        df_numeric = df
    
    outliers = {}
    total_outliers = 0
    
    if method == 'zscore':
        # Z-score method
        from scipy import stats
        z_scores = np.abs(stats.zscore(df_numeric))
        # Find where Z-scores exceed threshold
        for i, col in enumerate(df_numeric.columns):
            outlier_indices = np.where(z_scores[:, i] > threshold)[0]
            if len(outlier_indices) > 0:
                outliers[col] = {
                    'indices': outlier_indices.tolist(),
                    'values': df_numeric.iloc[outlier_indices, i].tolist(),
                    'count': len(outlier_indices)
                }
                total_outliers += len(outlier_indices)
    
    elif method == 'iqr':
        # IQR method
        for col in df_numeric.columns:
            Q1 = df_numeric[col].quantile(0.25)
            Q3 = df_numeric[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Find outliers based on IQR
            outlier_indices = df_numeric[(df_numeric[col] < lower_bound) | 
                                         (df_numeric[col] > upper_bound)].index.tolist()
            if len(outlier_indices) > 0:
                outliers[col] = {
                    'indices': outlier_indices,
                    'values': df_numeric.loc[outlier_indices, col].tolist(),
                    'count': len(outlier_indices)
                }
                total_outliers += len(outlier_indices)
    
    # Create a summary
    summary = {
        'method': method,
        'threshold': threshold,
        'total_outliers': total_outliers,
        'percentage_outliers': f"{(total_outliers / (len(df_numeric) * len(df_numeric.columns)) * 100):.2f}%",
        'affected_features': len(outliers)
    }
    
    # Create a visualization of the outliers
    plt.figure(figsize=(12, 8))
    
    # Select key measurements for visualization
    key_measurements = ['Head circumference', 'Head breadth', 'Head length', 'Total head height']
    key_measurements = [m for m in key_measurements if m in df_numeric.columns]
    
    if len(key_measurements) > 0:
        # Create boxplots to visualize the distributions and outliers
        plt.boxplot([df_numeric[col] for col in key_measurements], 
                   labels=key_measurements, 
                   flierprops=dict(marker='o', markerfacecolor='red', markersize=10))
        plt.title(f'Outlier Detection: {method.upper()} Method', fontsize=14)
        plt.ylabel('Measurement (mm)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        
        # Save as base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=80)
        buffer.seek(0)
        outlier_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
    else:
        outlier_plot = None
    
    # Create a DataFrame for all outliers
    outlier_rows = []
    for feature, data in outliers.items():
        for idx, val in zip(data['indices'], data['values']):
            outlier_rows.append({
                'Feature': feature,
                'Index': idx,
                'Value': val,
                'Method': method,
                'Threshold': threshold
            })
    
    outliers_df = pd.DataFrame(outlier_rows) if outlier_rows else pd.DataFrame()
    
    return outliers_df, outlier_plot, summary

def perform_pca(df, n_components=5):
    """Perform PCA on the anthropometric data"""
    # Remove gender column if it exists
    if 'Gender' in df.columns:
        data_for_pca = df.drop('Gender', axis=1)
    else:
        data_for_pca = df
        
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_for_pca)
    
    # Apply PCA
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)
    
    # Create PCA dataframe
    pca_df = pd.DataFrame(
        data=pca_result[:, :n_components],  # Take first n principal components
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    # Add gender column back if it existed
    if 'Gender' in df.columns:
        pca_df['Gender'] = df['Gender'].values
    
    # Calculate variance explained
    variance_explained = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_explained)
    
    # Create component loadings dataframe
    loadings = pd.DataFrame(
        data=pca.components_[:n_components].T,  # Transpose to get features in rows
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=data_for_pca.columns
    )
    
    return pca_df, variance_explained, cumulative_variance, loadings

def generate_plots(df, pca_df, variance_explained, cumulative_variance):
    """Generate plots for data visualization"""
    plots = {}
    
    # 1. Distribution of key measurements
    key_measurements = ['Total head height', 'Head circumference', 'Face length', 'Nose length', 'Bizygomatic breadth']
    
    # Check if we have gender data
    if 'Gender' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 7))
        male_data = df[df['Gender'] == 'Male'][key_measurements]
        female_data = df[df['Gender'] == 'Female'][key_measurements]
        
        # Create positions for grouped boxplots
        positions1 = np.array([1, 3, 5, 7, 9])
        positions2 = np.array([2, 4, 6, 8, 10])
        
        # Create box plots for male and female data
        bplot1 = ax.boxplot(male_data.values, positions=positions1, patch_artist=True,
                          boxprops=dict(facecolor='lightblue'), widths=0.6)
        bplot2 = ax.boxplot(female_data.values, positions=positions2, patch_artist=True,
                          boxprops=dict(facecolor='lightpink'), widths=0.6)
        
        # Set labels and title
        ax.set_xticks(np.arange(1.5, 11, 2))
        ax.set_xticklabels(key_measurements, rotation=45, ha='right')
        ax.set_ylabel('Millimeters')
        ax.set_title('Distribution of Key Anthropometric Measurements by Gender')
        
        # Add legend
        ax.legend([bplot1["boxes"][0], bplot2["boxes"][0]], ['Male', 'Female'], loc='upper right')
    else:
        plt.figure(figsize=(10, 6))
        df[key_measurements].boxplot()
        plt.title('Distribution of Key Anthropometric Measurements')
        plt.ylabel('Millimeters')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots['boxplot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # 2. Scree plot for PCA
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(variance_explained) + 1), variance_explained)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
    plt.title('Scree Plot')
    plt.xlabel('Principal Components')
    plt.ylabel('Proportion of Variance Explained')
    plt.xticks(range(1, min(11, len(variance_explained) + 1)))
    plt.axhline(y=0.8, color='r', linestyle='-')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots['scree_plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # 3. First two principal components scatter plot
    plt.figure(figsize=(10, 6))
    
    # Check if we have gender data for PCA plot
    if 'Gender' in pca_df.columns:
        male_pca = pca_df[pca_df['Gender'] == 'Male']
        female_pca = pca_df[pca_df['Gender'] == 'Female']
        
        plt.scatter(male_pca['PC1'], male_pca['PC2'], c='blue', label='Male', alpha=0.7)
        plt.scatter(female_pca['PC1'], female_pca['PC2'], c='red', label='Female', alpha=0.7)
        plt.legend()
        plt.title('PCA: First Two Principal Components by Gender')
    else:
        plt.scatter(pca_df['PC1'], pca_df['PC2'])
        plt.title('PCA: First Two Principal Components')
        
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots['pca_scatter'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # 4. Gender comparison plot (if gender data is available)
    if 'Gender' in df.columns:
        # Create a comparison plot for average measurements
        male_means = df[df['Gender'] == 'Male'][key_measurements].mean()
        female_means = df[df['Gender'] == 'Female'][key_measurements].mean()
        
        comparison_df = pd.DataFrame({
            'Male': male_means,
            'Female': female_means
        })
        
        plt.figure(figsize=(12, 6))
        ax = comparison_df.plot(kind='bar', figsize=(12, 6))
        plt.title('Average Measurements by Gender')
        plt.ylabel('Millimeters')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Gender')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plots['gender_comparison'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
    
    return plots

# ----- TACTICAL HELMET DESIGN FUNCTIONS -----

def analyze_helmet_parameters():
    """Analyze key parameters for tactical helmet design following MIL-STD-1472"""
    
    # Generate male and female data
    male_df = create_dataframe_male()
    female_df = create_dataframe_female()
    
    # Define critical helmet design parameters based on MIL-STD-1472
    # These parameters are critical for design accommodation per section 5.6.3
    helmet_parameters = [
        "Head circumference",
        "Head breadth",
        "Head length",
        "Total head height",
        "Bitragion breadth",
        "Frontotemporale breadth",
        "Bizygomatic breadth",
        "Inion to vertex height",
        "Sagittal arc",
        "Bitragion arc"
    ]
    
    # Filter for the parameters we have in our dataset (some may have slightly different names)
    available_parameters = [param for param in helmet_parameters if param in male_df.columns]
    
    # Calculate the 5th, 50th, and 95th percentiles for each gender
    # MIL-STD-1472 requires design accommodation from 5th to 95th percentile (section 4.4.1)
    male_percentiles = male_df[available_parameters].quantile([0.05, 0.50, 0.95]).round(2)
    female_percentiles = female_df[available_parameters].quantile([0.05, 0.50, 0.95]).round(2)
    
    # Calculate the range for design accommodation (5th percentile female to 95th percentile male)
    # This follows MIL-STD-1472 Section 4.4.2's requirement for full population accommodation
    design_min = female_percentiles.loc[0.05].copy()
    design_max = male_percentiles.loc[0.95].copy()
    
    # Calculate the design range
    design_range = pd.DataFrame({
        'Min (5th %ile Female)': design_min,
        'Max (95th %ile Male)': design_max,
        'Range (mm)': design_max - design_min,
        'Range (%)': ((design_max - design_min) / design_min * 100).round(1)
    })
    
    # Create adjustment factors based on MIL-STD-1472 section 5.6.4 
    # (equipment dimensions should accommodate added bulk from clothing, gear)
    # For helmet design, add clearance for thermal comfort, comms equipment
    clearance_factors = {
        "Head circumference": 1.03,  # 3% for comfort padding
        "Head breadth": 1.02,        # 2% for side padding
        "Head length": 1.02,         # 2% for front/back padding
        "Total head height": 1.05,   # 5% for top padding and suspension
        "Bitragion breadth": 1.10,   # 10% for communications equipment
        "Frontotemporale breadth": 1.02, # 2% for comfort
        "Bizygomatic breadth": 1.02, # 2% for comfort
        "Inion to vertex height": 1.03, # 3% for comfort
        "Sagittal arc": 1.03,        # 3% for suspension system
        "Bitragion arc": 1.10        # 10% for comms and suspension
    }
    
    # Apply clearance factors to maximum values
    design_specs = design_range.copy()
    for param in available_parameters:
        if param in clearance_factors:
            design_specs.loc[param, 'Design Max (with clearance)'] = design_range.loc[param, 'Max (95th %ile Male)'] * clearance_factors[param]
    
    # Create size categories based on MIL-STD-1472 section 5.6.3.5 (size ranges)
    # Typically 3-5 sizes are recommended for headgear
    # We'll use Head Circumference as the primary sizing parameter (most common practice)
    size_ranges = None
    size_coverage_df = None
    
    if "Head circumference" in available_parameters:
        hc_min = design_range.loc["Head circumference", "Min (5th %ile Female)"]
        hc_max = design_specs.loc["Head circumference", "Design Max (with clearance)"]
        hc_range = hc_max - hc_min
        
        # Create 5 sizes (XS, S, M, L, XL) per MIL-STD-1472
        size_ranges = pd.DataFrame({
            "Size": ["XS", "S", "M", "L", "XL"],
            "Min (mm)": np.linspace(hc_min, hc_max - hc_range/5, 5).round(1),
            "Max (mm)": np.linspace(hc_min + hc_range/5, hc_max, 5).round(1),
        })
        
        # Calculate population coverage for each size
        male_hc = male_df["Head circumference"]
        female_hc = female_df["Head circumference"]
        size_coverage = []
        
        for i, row in size_ranges.iterrows():
            size_min, size_max = row["Min (mm)"], row["Max (mm)"]
            
            # Count males and females in this size range
            males_in_range = ((male_hc >= size_min) & (male_hc < size_max)).sum()
            females_in_range = ((female_hc >= size_min) & (female_hc < size_max)).sum()
            
            # Calculate percentages
            male_pct = males_in_range / len(male_hc) * 100
            female_pct = females_in_range / len(female_hc) * 100
            total_pct = (males_in_range + females_in_range) / (len(male_hc) + len(female_hc)) * 100
            
            size_coverage.append({
                "Size": row["Size"],
                "Male %": round(male_pct, 1),
                "Female %": round(female_pct, 1),
                "Total %": round(total_pct, 1)
            })
        
        size_coverage_df = pd.DataFrame(size_coverage)
    
    # Generate detailed report
    print("=" * 80)
    print("TACTICAL HELMET DESIGN PARAMETERS ANALYSIS")
    print("Based on MIL-STD-1472 Human Engineering Standards")
    print("=" * 80)
    
    print("\nKEY ANTHROPOMETRIC PARAMETERS (all measurements in mm):")
    print("-" * 80)
    print(design_range)
    
    print("\nDESIGN SPECIFICATIONS WITH CLEARANCE:")
    print("-" * 80)
    design_specs_display = design_specs[['Min (5th %ile Female)', 'Max (95th %ile Male)', 'Design Max (with clearance)']]
    print(design_specs_display)
    
    if size_ranges is not None:
        print("\nHELMET SIZE CATEGORIES (Head Circumference):")
        print("-" * 80)
        print(size_ranges[["Size", "Min (mm)", "Max (mm)"]])
        
        print("\nPOPULATION COVERAGE BY SIZE:")
        print("-" * 80)
        print(size_coverage_df)
    
    # Generate plots
    output_dir = "helmet_design_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Create a size distribution chart 
    plt.figure(figsize=(12, 8))
    
    # Male distribution
    plt.hist(male_df["Head circumference"], bins=30, alpha=0.6, color=MILITARY_GREEN, 
             label='Male', density=True)
    
    # Female distribution
    plt.hist(female_df["Head circumference"], bins=30, alpha=0.6, color=MILITARY_YELLOW,
             label='Female', density=True)
    
    # Add size range markers
    if "Head circumference" in available_parameters and size_ranges is not None:
        colors = ['#FF9999', '#FFCC99', '#FFFF99', '#CCFF99', '#99FFCC']
        
        for i, row in size_ranges.iterrows():
            plt.axvline(x=row["Min (mm)"], color='white', linestyle='--', alpha=0.5)
            plt.axvspan(row["Min (mm)"], row["Max (mm)"], alpha=0.2, color=colors[i], 
                       label=f'Size {row["Size"]}')
    
    plt.title("Head Circumference Distribution with Helmet Size Ranges", fontsize=16)
    plt.xlabel("Head Circumference (mm)", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add MIL-STD-1472 annotation
    if "Head circumference" in available_parameters:
        plt.annotate("MIL-STD-1472 requires design\naccommodation from 5th to 95th percentile",
                    xy=(hc_min, 0.003), xytext=(hc_min-30, 0.006),
                    arrowprops=dict(facecolor='white', shrink=0.05, width=2),
                    bbox=dict(boxstyle="round,pad=0.5", fc=MILITARY_RED, alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/helmet_size_distribution.png", dpi=300)
    plt.close()
    
    # 2. Create a radar chart comparing male vs female dimensions
    # Select key parameters for the radar chart
    radar_params = available_parameters[:6]  # Take the top 6 parameters for readability
    
    # Normalize data for radar chart
    male_median = male_df[radar_params].median()
    female_median = female_df[radar_params].median()
    
    # Set up radar chart
    angles = np.linspace(0, 2*np.pi, len(radar_params), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Add normalized data
    male_values = male_median.tolist()
    male_values += male_values[:1]  # Close the loop
    
    female_values = female_median.tolist()
    female_values += female_values[:1]  # Close the loop
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Normalize data for better visualization
    max_values = np.maximum(male_median, female_median)
    male_norm = male_median / max_values
    female_norm = female_median / max_values
    
    male_norm_values = male_norm.tolist()
    male_norm_values += male_norm_values[:1]
    
    female_norm_values = female_norm.tolist()
    female_norm_values += female_norm_values[:1]
    
    # Plot data
    ax.plot(angles, male_norm_values, color=MILITARY_GREEN, linewidth=2, label='Male (median)')
    ax.fill(angles, male_norm_values, color=MILITARY_GREEN, alpha=0.25)
    
    ax.plot(angles, female_norm_values, color=MILITARY_YELLOW, linewidth=2, label='Female (median)')
    ax.fill(angles, female_norm_values, color=MILITARY_YELLOW, alpha=0.25)
    
    # Add parameter labels
    plt.xticks(angles[:-1], radar_params, size=14)
    
    # Add a title
    plt.title('Male vs Female Head Dimensions (Normalized)', size=16)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gender_comparison_radar.png", dpi=300)
    plt.close()
    
    # 3. Create a bar chart showing the design ranges
    plt.figure(figsize=(14, 8))
    
    # Extract data for plotting
    params = design_range.index
    design_mins = design_range['Min (5th %ile Female)']
    design_maxes = design_range['Max (95th %ile Male)']
    
    # Plot
    x = np.arange(len(params))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.barh(x - width/2, design_mins, width, label='5th %ile Female', color=MILITARY_YELLOW)
    rects2 = ax.barh(x + width/2, design_maxes, width, label='95th %ile Male', color=MILITARY_GREEN)
    
    # Add MIL-STD-1472 design range indicators
    for i, param in enumerate(params):
        if param in design_specs.index and 'Design Max (with clearance)' in design_specs.columns:
            design_max_with_clearance = design_specs.loc[param, 'Design Max (with clearance)']
            ax.plot([design_max_with_clearance], [i + width/2], marker='*', 
                   markersize=12, color=MILITARY_RED, label='_nolegend_')
            ax.hlines(y=i, xmin=design_mins.iloc[i], xmax=design_max_with_clearance, 
                     colors=MILITARY_RED, linestyles='--', linewidth=2, label='_nolegend_')
    
    # Add labels and formatting
    ax.set_xlabel('Measurement (mm)', fontsize=14)
    ax.set_title('Design Range for Tactical Helmet Parameters', fontsize=16)
    ax.set_yticks(x)
    ax.set_yticklabels(params)
    ax.legend()
    
    # Add annotation for MIL-STD-1472
    ax.annotate('* Red stars indicate max design values\nwith MIL-STD-1472 clearances applied',
               xy=(0.7, 0.05), xycoords='axes fraction',
               bbox=dict(boxstyle="round,pad=0.5", fc=MILITARY_RED, alpha=0.8),
               fontsize=12, color='white')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/design_range_chart.png", dpi=300)
    plt.close()
    
    # Save all data to Excel file
    with pd.ExcelWriter(f"{output_dir}/tactical_helmet_specifications.xlsx") as writer:
        design_range.to_excel(writer, sheet_name='Design Range')
        design_specs.to_excel(writer, sheet_name='Design Specs with Clearance')
        
        if size_ranges is not None and size_coverage_df is not None:
            size_ranges.to_excel(writer, sheet_name='Size Categories', index=False)
            size_coverage_df.to_excel(writer, sheet_name='Population Coverage', index=False)
        
        male_percentiles.to_excel(writer, sheet_name='Male Percentiles')
        female_percentiles.to_excel(writer, sheet_name='Female Percentiles')
    
    print(f"\nAnalysis complete. Results saved to {output_dir}/")
    print(f"Design specifications file: {output_dir}/tactical_helmet_specifications.xlsx")
    print(f"Visualization charts saved as PNG files in {output_dir}/")

def check_helmet_fit(measurements, gender='male'):
    """
    Analyzes individual measurements to determine tactical helmet fit
    following MIL-STD-1472 Human Engineering requirements
    
    Parameters:
    -----------
    measurements : dict
        Dictionary of head measurements in mm
        e.g., {"Head circumference": 570, "Head breadth": 148, ...}
    gender : str
        'male' or 'female' - used for percentile analysis
    
    Returns:
    --------
    dict
        Fit assessment and recommended size
    """
    # Check if helmet design specs exist
    output_dir = "helmet_design_outputs"
    specs_file = f"{output_dir}/tactical_helmet_specifications.xlsx"
    
    if not os.path.exists(specs_file):
        print("Error: Helmet specifications not found. Please run helmet_design_parameters.py first.")
        raise FileNotFoundError("Helmet specifications not found. Please run the analysis first.")
    
    # Load design specs
    design_range = pd.read_excel(specs_file, sheet_name='Design Range', index_col=0)
    design_specs = pd.read_excel(specs_file, sheet_name='Design Specs with Clearance', index_col=0)
    
    # Load size categories
    try:
        size_ranges = pd.read_excel(specs_file, sheet_name='Size Categories')
    except:
        print("Warning: Size categories not found in the specifications file.")
        size_ranges = None
    
    # Load gender-specific percentile data
    if gender.lower() == 'male':
        percentiles = pd.read_excel(specs_file, sheet_name='Male Percentiles', index_col=0)
    else:
        percentiles = pd.read_excel(specs_file, sheet_name='Female Percentiles', index_col=0)
    
    # Analyze fit for each parameter
    fit_analysis = {}
    overall_percentiles = {}
    
    for param, value in measurements.items():
        if param in design_range.index:
            # Get design min and max
            design_min = design_range.loc[param, 'Min (5th %ile Female)']
            design_max = design_range.loc[param, 'Max (95th %ile Male)']
            
            # Get design max with clearance if available
            if 'Design Max (with clearance)' in design_specs.columns and param in design_specs.index:
                design_max_with_clearance = design_specs.loc[param, 'Design Max (with clearance)']
            else:
                design_max_with_clearance = design_max
            
            # Check if measurement is within design range
            if value < design_min:
                status = "BELOW RANGE"
                color = MILITARY_RED
            elif value > design_max_with_clearance:
                status = "ABOVE RANGE"
                color = MILITARY_RED
            elif value > design_max:
                status = "WITHIN CLEARANCE"
                color = MILITARY_YELLOW
            else:
                status = "WITHIN RANGE"
                color = MILITARY_GREEN
            
            # Calculate approximate percentile for this parameter
            if param in percentiles.columns:
                # The index labels might be 0.05, 0.5, 0.95 or '5th Percentile', '50th Percentile', '95th Percentile'
                # Let's check the index and use appropriate labels
                idx_names = percentiles.index.tolist()
                if 0.05 in idx_names:
                    p5 = percentiles.loc[0.05, param]
                    p50 = percentiles.loc[0.50, param]
                    p95 = percentiles.loc[0.95, param]
                elif '5th Percentile' in idx_names:
                    p5 = percentiles.loc['5th Percentile', param]
                    p50 = percentiles.loc['50th Percentile', param]
                    p95 = percentiles.loc['95th Percentile', param]
                else:
                    # Use the first, middle, and last index as fallback
                    p5 = percentiles.iloc[0, percentiles.columns.get_loc(param)]
                    p50 = percentiles.iloc[1, percentiles.columns.get_loc(param)]
                    p95 = percentiles.iloc[2, percentiles.columns.get_loc(param)]
                
                # Estimate percentile (rough linear interpolation)
                if value <= p5:
                    est_percentile = 5 * value / p5 if p5 > 0 else 0
                elif value <= p50:
                    est_percentile = 5 + (45 * (value - p5) / (p50 - p5)) if (p50 - p5) > 0 else 5
                elif value <= p95:
                    est_percentile = 50 + (45 * (value - p50) / (p95 - p50)) if (p95 - p50) > 0 else 50
                else:
                    est_percentile = 95 + (5 * (value - p95) / p95) if p95 > 0 else 100
                
                est_percentile = min(max(est_percentile, 0), 100)
                overall_percentiles[param] = est_percentile
            else:
                est_percentile = "Unknown"
            
            # Store analysis
            fit_analysis[param] = {
                'value': value,
                'design_min': design_min,
                'design_max': design_max,
                'design_max_clearance': design_max_with_clearance,
                'status': status,
                'color': color,
                'percentile': est_percentile
            }
    
    # Determine helmet size if head circumference is provided
    recommended_size = None
    size_fit = None
    
    if size_ranges is not None and "Head circumference" in measurements:
        hc = measurements["Head circumference"]
        
        for i, row in size_ranges.iterrows():
            if hc >= row["Min (mm)"] and hc < row["Max (mm)"]:
                recommended_size = row["Size"]
                # Check how well the size fits
                size_range = row["Max (mm)"] - row["Min (mm)"]
                position_in_range = (hc - row["Min (mm)"]) / size_range
                
                if position_in_range < 0.2:
                    size_fit = "Lower end of size range"
                elif position_in_range > 0.8:
                    size_fit = "Upper end of size range"
                else:
                    size_fit = "Good fit within size range"
                break
        
        # Handle edge cases
        if recommended_size is None:
            min_size = size_ranges["Size"].iloc[0]
            max_size = size_ranges["Size"].iloc[-1]
            
            if hc < size_ranges["Min (mm)"].iloc[0]:
                recommended_size = f"Below {min_size}"
                size_fit = "Too small for standard sizes"
            else:
                recommended_size = f"Above {max_size}"
                size_fit = "Too large for standard sizes"
    
    # Generate visualization of fit
    output_file = generate_fit_visualization(measurements, fit_analysis, recommended_size)
    
    # Return results
    return {
        'fit_analysis': fit_analysis,
        'recommended_size': recommended_size,
        'size_fit': size_fit,
        'visualization_file': output_file,
        'overall_percentiles': overall_percentiles
    }

def generate_fit_visualization(measurements, fit_analysis, recommended_size):
    """Generate a visualization of how the measurements fit within the design specs"""
    output_dir = "helmet_design_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Set up parameters and data
    params = list(fit_analysis.keys())
    values = [fit_analysis[param]['value'] for param in params]
    min_vals = [fit_analysis[param]['design_min'] for param in params]
    max_vals = [fit_analysis[param]['design_max'] for param in params]
    max_clearance = [fit_analysis[param]['design_max_clearance'] for param in params]
    colors = [fit_analysis[param]['color'] for param in params]
    
    # Plot
    y_pos = np.arange(len(params))
    
    # Plot design ranges as horizontal bars
    for i, param in enumerate(params):
        plt.barh(y_pos[i], max_clearance[i] - min_vals[i], left=min_vals[i], 
                height=0.5, color='lightgray', alpha=0.3)
        
        # Add marker for standard range vs clearance
        plt.axvline(x=max_vals[i], ymin=(i-0.25)/len(params), ymax=(i+0.25)/len(params), 
                   color='gray', linestyle='--')
    
    # Plot measured values as points
    for i, (param, value) in enumerate(zip(params, values)):
        plt.scatter(value, y_pos[i], color=colors[i], s=100, zorder=5)
        
        # Add value labels
        plt.text(value, y_pos[i], f" {value:.1f}", 
                va='center', ha='left' if value < np.mean([min_vals[i], max_clearance[i]]) else 'right',
                color=colors[i], fontweight='bold')
    
    # Add labels and formatting
    plt.yticks(y_pos, params)
    plt.xlabel('Measurement (mm)', fontsize=12)
    plt.title('Individual Measurements vs Tactical Helmet Design Range', fontsize=16)
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=MILITARY_GREEN, markersize=10, label='Within Range'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=MILITARY_YELLOW, markersize=10, label='Within Clearance'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=MILITARY_RED, markersize=10, label='Outside Range')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    # Add size recommendation if available
    if recommended_size is not None:
        plt.annotate(f'Recommended Size: {recommended_size}',
                    xy=(0.02, 0.97), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", fc=MILITARY_GREEN, alpha=0.8),
                    fontsize=14, color='white')
    
    plt.tight_layout()
    
    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = f"{output_dir}/individual_helmet_fit_{timestamp}.png"
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    return output_file

# ----- FLASK ROUTES -----

@app.route('/')
def index():
    """Render the main page with data analysis"""
    # Get gender selection from query parameter, default to 'both'
    gender = request.args.get('gender', 'both')
    
    # Create dataframe based on gender selection
    df = create_dataframe(gender)
    
    # Perform PCA
    pca_df, variance_explained, cumulative_variance, loadings = perform_pca(df)
    
    # Calculate statistics (exclude Gender column if it exists)
    if 'Gender' in df.columns:
        df_numeric = df.drop('Gender', axis=1)
    else:
        df_numeric = df
        
    # Calculate percentiles
    percentiles = df_numeric.quantile([0.05, 0.50, 0.95]).round(3)
    percentiles.index = ['5th Percentile', '50th Percentile', '95th Percentile']
    
    # Generate plots
    plots = generate_plots(df, pca_df, variance_explained, cumulative_variance)
    
    # Calculate summary statistics
    summary = df_numeric.describe().round(3)
    
    # Convert dataframes to HTML for display
    summary_html = summary.to_html(classes="table table-striped table-hover")
    percentiles_html = percentiles.to_html(classes="table table-striped table-hover")
    
    # Select top loadings for display
    top_loadings = {}
    for i in range(5):
        pc_name = f'PC{i+1}'
        # Get top 5 measurements with highest absolute loadings
        top_indices = loadings[pc_name].abs().nlargest(5).index
        top_loadings[pc_name] = loadings.loc[top_indices, pc_name].to_dict()
    
    return render_template('index.html', 
                          summary_html=summary_html, 
                          percentiles_html=percentiles_html,
                          plots=plots,
                          variance_explained=[round(v*100, 2) for v in variance_explained[:5]],
                          cumulative_variance=[round(v*100, 2) for v in cumulative_variance[:5]],
                          top_loadings=top_loadings,
                          selected_gender=gender)

@app.route('/download')
def download():
    """Generate and download Excel file with analysis results"""
    # Get gender selection from query parameter, default to 'both'
    gender = request.args.get('gender', 'both')
    
    if gender == 'both':
        # Create male and female dataframes separately for the Excel report
        male_df = create_dataframe_male()
        female_df = create_dataframe_female()
        
        # Perform PCA on each dataset
        male_pca_df, male_variance, male_cumulative, male_loadings = perform_pca(male_df)
        female_pca_df, female_variance, female_cumulative, female_loadings = perform_pca(female_df)
        
        # Combined dataset for overall analysis
        combined_df = create_dataframe('both')
        combined_pca_df, combined_variance, combined_cumulative, combined_loadings = perform_pca(combined_df)
        
        # Calculate statistics for each dataset
        # No need to check for Gender column since these are single-gender datasets
        male_percentiles = male_df.quantile([0.05, 0.50, 0.95]).round(3)
        male_percentiles.index = ['5th Percentile', '50th Percentile', '95th Percentile']
        male_summary = male_df.describe().round(3)
        
        female_percentiles = female_df.quantile([0.05, 0.50, 0.95]).round(3)
        female_percentiles.index = ['5th Percentile', '50th Percentile', '95th Percentile']
        female_summary = female_df.describe().round(3)
        
        # For combined dataframe, drop Gender column for calculations
        combined_df_numeric = combined_df.drop('Gender', axis=1)
        
        # Calculate gender differences
        mean_diff = (male_df.mean() - female_df.mean()).round(3)
        mean_diff = pd.DataFrame(mean_diff, columns=['Male-Female Difference'])
        
        # Create Excel writer object
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Write male data
            male_df.to_excel(writer, sheet_name='Male Raw Data', index=False)
            male_summary.to_excel(writer, sheet_name='Male Summary Stats')
            male_percentiles.to_excel(writer, sheet_name='Male Percentiles')
            male_pca_df.to_excel(writer, sheet_name='Male PCA Results', index=False)
            
            # Write female data
            female_df.to_excel(writer, sheet_name='Female Raw Data', index=False)
            female_summary.to_excel(writer, sheet_name='Female Summary Stats')
            female_percentiles.to_excel(writer, sheet_name='Female Percentiles')
            female_pca_df.to_excel(writer, sheet_name='Female PCA Results', index=False)
            
            # Write combined data
            combined_df.to_excel(writer, sheet_name='Combined Raw Data', index=False)
            mean_diff.to_excel(writer, sheet_name='Gender Differences')
            
            # Write PCA loadings
            male_loadings.to_excel(writer, sheet_name='Male PCA Loadings')
            female_loadings.to_excel(writer, sheet_name='Female PCA Loadings')
        
        # Set up the response
        output.seek(0)
        return send_file(output, 
                         download_name=f'anthropometric_data_male_female_comparison.xlsx',
                         as_attachment=True,
                         mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    else:
        # Create dataframe for the selected gender
        df = create_dataframe(gender)
        
        # Perform PCA
        pca_df, variance_explained, cumulative_variance, loadings = perform_pca(df)
        
        # Calculate statistics
        if 'Gender' in df.columns:  # Should not happen in this case, but check anyway
            df = df.drop('Gender', axis=1)
            
        percentiles = df.quantile([0.05, 0.50, 0.95]).round(3)
        percentiles.index = ['5th Percentile', '50th Percentile', '95th Percentile']
        summary = df.describe().round(3)
        
        # Create Excel writer object
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Raw Data', index=False)
            summary.to_excel(writer, sheet_name='Summary Statistics')
            percentiles.to_excel(writer, sheet_name='Percentiles')
            pca_df.to_excel(writer, sheet_name='PCA Results', index=False)
            loadings.to_excel(writer, sheet_name='PCA Loadings')
        
        # Set up the response
        output.seek(0)
        return send_file(output, 
                         download_name=f'anthropometric_data_{gender}.xlsx',
                         as_attachment=True,
                         mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

@app.route('/api/summary')
def api_summary():
    """API endpoint for summary statistics"""
    # Get gender selection from query parameter, default to 'both'
    gender = request.args.get('gender', 'both')
    
    # Create dataframe based on gender selection
    df = create_dataframe(gender)
    
    # Remove gender column if present
    if 'Gender' in df.columns:
        df = df.drop('Gender', axis=1)
    
    # Get summary statistics
    result = {
        'gender': gender,
        'statistics': df.describe().round(3).to_dict()
    }
    
    # Add gender comparison if both genders are requested
    if gender == 'both':
        male_df = create_dataframe_male()
        female_df = create_dataframe_female()
        mean_diff = (male_df.mean() - female_df.mean()).round(3)
        result['gender_differences'] = mean_diff.to_dict()
    
    return jsonify(result)

@app.route('/advanced_analysis')
def advanced_analysis():
    """Render the advanced data analysis page with additional features"""
    # Get gender selection from query parameter, default to 'both'
    gender = request.args.get('gender', 'both')
    analysis_type = request.args.get('analysis_type', 'correlation')
    
    # Create dataframe based on gender selection
    df = create_dataframe(gender)
    
    analysis_results = {}
    
    if analysis_type == 'correlation':
        # Perform correlation analysis
        correlation_matrix, correlation_plot, key_insights = calculate_correlation_analysis(df)
        analysis_results = {
            'title': 'Correlation Analysis',
            'description': 'Analysis of relationships between different anthropometric measurements',
            'plot': correlation_plot,
            'insights': key_insights,
            'matrix_html': correlation_matrix.to_html(classes="table table-striped table-hover table-sm")
        }
    
    elif analysis_type == 'frequency':
        # Perform frequency distribution analysis
        key_measurements = ['Head circumference', 'Head breadth', 'Head length', 'Total head height']
        histograms, summary = perform_frequency_analysis(df, key_measurements)
        analysis_results = {
            'title': 'Frequency Distribution Analysis',
            'description': 'Detailed analysis of the distribution of key measurements',
            'plots': histograms,
            'summary': summary,
            'measurements': key_measurements
        }
    
    elif analysis_type == 'clusters':
        # Perform cluster analysis
        n_clusters = int(request.args.get('n_clusters', 3))
        cluster_df, cluster_plot, cluster_stats = perform_cluster_analysis(df, n_clusters)
        analysis_results = {
            'title': 'Cluster Analysis',
            'description': f'K-means clustering with {n_clusters} clusters to identify natural groupings',
            'plot': cluster_plot,
            'stats': cluster_stats,
            'n_clusters': n_clusters
        }
    
    elif analysis_type == 'outliers':
        # Perform outlier detection
        method = request.args.get('method', 'zscore')
        threshold = float(request.args.get('threshold', 3.0))
        outliers_df, outlier_plot, summary = detect_outliers(df, method, threshold)
        analysis_results = {
            'title': 'Outlier Detection',
            'description': f'Detection of outliers using {method.upper()} method',
            'plot': outlier_plot,
            'summary': summary,
            'method': method,
            'threshold': threshold
        }
        if not outliers_df.empty:
            analysis_results['outliers_html'] = outliers_df.to_html(classes="table table-striped table-hover table-sm")
        else:
            analysis_results['outliers_html'] = '<p>No outliers detected with the current settings.</p>'
    
    return render_template('advanced_analysis.html',
                        analysis_results=analysis_results,
                        analysis_type=analysis_type,
                        selected_gender=gender)

@app.route('/api/gender_comparison')
def gender_comparison():
    """API endpoint for gender comparison"""
    # Get key measurements of interest
    measurements = request.args.getlist('measurements')
    if not measurements:
        # Default measurements if none specified
        measurements = ['Total head height', 'Head circumference', 'Face length', 'Nose length', 'Bizygomatic breadth']
    
    male_df = create_dataframe_male()
    female_df = create_dataframe_female()
    
    # Filter to requested measurements
    valid_measurements = [m for m in measurements if m in male_df.columns]
    
    if not valid_measurements:
        return jsonify({'error': 'No valid measurements provided'}), 400
    
    # Calculate statistics for the requested measurements
    male_stats = male_df[valid_measurements].describe().round(3)
    female_stats = female_df[valid_measurements].describe().round(3)
    
    # Calculate differences
    mean_diff = (male_df[valid_measurements].mean() - female_df[valid_measurements].mean()).round(3)
    
    result = {
        'measurements': valid_measurements,
        'male_statistics': male_stats.to_dict(),
        'female_statistics': female_stats.to_dict(),
        'differences': mean_diff.to_dict()
    }
    
    return jsonify(result)

@app.route('/helmet_design')
def helmet_design():
    """Render the helmet design analysis page"""
    # Check if design specs file exists
    specs_file = os.path.join('helmet_design_outputs', 'tactical_helmet_specifications.xlsx')
    specs_exist = os.path.exists(specs_file)
    
    # Get the list of generated image files
    image_files = []
    if specs_exist:
        for file in os.listdir('helmet_design_outputs'):
            if file.endswith('.png') and not file.startswith('individual_helmet_fit'):
                image_files.append(file)
    
    # Render template with available data
    return render_template('helmet_design.html', 
                          specs_exist=specs_exist,
                          image_files=image_files)

@app.route('/run_helmet_analysis', methods=['POST'])
def run_helmet_analysis():
    """Run the helmet design parameter analysis"""
    try:
        # Clear any existing image files except individual fit files
        for file in os.listdir('helmet_design_outputs'):
            if file.endswith('.png') and not file.startswith('individual_helmet_fit'):
                os.remove(os.path.join('helmet_design_outputs', file))
        
        # Run the analysis
        analyze_helmet_parameters()
        
        # Redirect back to the helmet design page
        return redirect(url_for('helmet_design'))
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/individual_fit')
def individual_fit():
    """Render the individual helmet fit analyzer page"""
    # Check if design specs file exists (required for fit analysis)
    specs_file = os.path.join('helmet_design_outputs', 'tactical_helmet_specifications.xlsx')
    if not os.path.exists(specs_file):
        # If specs don't exist, redirect to helmet design page
        return render_template('individual_fit.html', 
                              specs_exist=False,
                              analysis_result=None,
                              fit_image=None)
    
    return render_template('individual_fit.html', 
                          specs_exist=True,
                          analysis_result=None,
                          fit_image=None)

@app.route('/analyze_fit', methods=['POST'])
def analyze_fit():
    """Analyze an individual's measurements for helmet fit"""
    try:
        # Get input measurements from form
        measurements = {}
        gender = request.form.get('gender', 'male')
        
        # Important parameters for helmet design
        helmet_params = [
            'Head circumference', 'Head breadth', 'Head length', 
            'Total head height', 'Bitragion breadth', 'Frontotemporale breadth',
            'Bizygomatic breadth', 'Inion to vertex height', 'Sagittal arc', 'Bitragion arc'
        ]
        
        # Parse measurements from form
        for param in helmet_params:
            if param in request.form and request.form[param].strip():
                try:
                    measurements[param] = float(request.form[param])
                except ValueError:
                    return render_template('error.html', 
                                         error=f"Invalid value for {param}. Please enter a number.")
        
        if not measurements:
            return render_template('error.html', 
                                 error="No measurements provided. Please enter at least one measurement.")
        
        # Run the fit analysis
        result = check_helmet_fit(measurements, gender)
        
        # Get the image path for display
        fit_image = os.path.basename(result['visualization_file'])
        
        # Render the template with results
        return render_template('individual_fit.html', 
                             specs_exist=True,
                             analysis_result=result,
                             fit_image=fit_image)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/design_files/<filename>')
def design_files(filename):
    """Serve files from the helmet design outputs directory"""
    return send_file(os.path.join('helmet_design_outputs', filename))

@app.route('/download_specifications')
def download_specifications():
    """Download the helmet design specifications Excel file"""
    specs_file = os.path.join('helmet_design_outputs', 'tactical_helmet_specifications.xlsx')
    
    # Check if the file exists
    if not os.path.exists(specs_file):
        return render_template('error.html', 
                              error="Design specifications not found. Please run the analysis first.")
    
    return send_file(specs_file,
                    as_attachment=True,
                    download_name='tactical_helmet_specifications.xlsx')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)