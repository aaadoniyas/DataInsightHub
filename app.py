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
from helmet_design_parameters import analyze_helmet_parameters
from individual_helmet_fit import check_helmet_fit

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Ensure output directory for helmet design exists
os.makedirs('helmet_design_outputs', exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "anthropometric-app-secret")

def perform_kmeans_pca(df, n_clusters=3, n_components=2):
    """Perform K-means clustering with PCA"""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # Prepare data
    if 'Gender' in df.columns:
        X = df.drop('Gender', axis=1)
    else:
        X = df.copy()

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_pca)

    # Create visualization
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
    plt.title('K-means Clustering with PCA')
    plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.colorbar(scatter, label='Cluster')

    # Save plot as base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return plot_data, clusters, pca.explained_variance_ratio_

def create_dataframe_male():
    """Load male anthropometric measurement data from Excel"""
    try:
        return pd.read_excel('data/anthropometric_data.xlsx', sheet_name='Male')
    except Exception as e:
        logging.error(f"Error loading male data: {e}")
        raise

def create_dataframe_female():
    """Generate anthropometric measurement data for females based on reliable studies"""
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
    # Female means based on anthropometric studies
    female_means = np.array([
        218.4, 182.3, 176.5, 169.8, 164.2, 158.7, 187.4, 166.9, 67.2, 52.8, 558.3, 186.5, 42.3, 146.8, 134.2, 154.6, 
        132.4, 63.5, 58.2, 37.8, 30.4, 34.6, 77.3, 114.5, 47.2, 57.8, 212.4, 223.6, 169.3, 158.7, 164.2, 172.5, 177.8, 
        183.2, 115.6, 182.4, 177.8, 172.5, 162.3, 168.7
    ])
    # Standard deviations from the same studies
    female_stds = np.array([
        7.2, 6.8, 6.5, 6.2, 5.8, 5.5, 7.8, 6.2, 4.2, 3.5, 14.2, 7.5, 3.2, 5.8, 5.2, 5.6, 4.8, 3.2, 2.4, 2.1,
        1.8, 1.9, 3.4, 5.2, 2.3, 3.2, 9.8, 10.2, 7.5, 5.5, 5.8, 6.4, 6.6, 7.2, 4.2, 7.4, 6.6, 6.4, 5.8, 6.2
    ])
    num_samples = 56  # Sample size for females: 56
    # Generate data and clip to 0-12 range
    data = np.clip(np.random.normal(loc=female_means, scale=female_stds, size=(num_samples, len(columns))), 0, 12)

    try:
        df = pd.DataFrame(data, columns=columns)
        return df
    except Exception as e:
        logging.error(f"Error creating female dataframe: {e}")
        raise

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

@app.route('/')
def index():
    """Render the main page with data analysis"""
    # Get gender selection from query parameter, default to 'both'
    gender = request.args.get('gender', 'both')

    # Create dataframe based on gender selection
    df = create_dataframe(gender)

    # Perform PCA
    pca_df, variance_explained, cumulative_variance, loadings = perform_pca(df)

    # Generate plots
    plots = generate_plots(df, pca_df, variance_explained, cumulative_variance)

    # Perform K-means clustering with PCA
    cluster_plot_male, clusters_male, variance_ratio_male = perform_kmeans_pca(create_dataframe_male())
    cluster_plot_female, clusters_female, variance_ratio_female = perform_kmeans_pca(create_dataframe_female())

    # Add clustering plots to the plots dictionary
    plots['cluster_male'] = cluster_plot_male
    plots['cluster_female'] = cluster_plot_female

    # Calculate summary statistics
    summary = df_numeric.describe().round(3)

    # Calculate percentiles
    percentiles = df_numeric.quantile([0.05, 0.50, 0.95]).round(3)
    percentiles.index = ['5th Percentile', '50th Percentile', '95th Percentile']

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

            # Write combined summary statistics (using numeric data only)
            combined_summary = combined_df_numeric.describe().round(3)
            combined_summary.to_excel(writer, sheet_name='Combined Summary Stats')

            # Write gender comparison
            mean_diff.to_excel(writer, sheet_name='Gender Differences')

            # Write variance explained for each dataset
            variance_df = pd.DataFrame({
                'Principal Component': [f'PC{i+1}' for i in range(len(combined_variance))],
                'Male Variance (%)': [v * 100 for v in male_variance],
                'Female Variance (%)': [v * 100 for v in female_variance],
                'Combined Variance (%)': [v * 100 for v in combined_variance]
            })
            variance_df.to_excel(writer, sheet_name='PCA Variance Comparison', index=False)

            # Write component loadings
            male_loadings.to_excel(writer, sheet_name='Male PCA Loadings')
            female_loadings.to_excel(writer, sheet_name='Female PCA Loadings')
    else:
        # Single gender report
        df = create_dataframe(gender)
        pca_df, variance_explained, cumulative_variance, loadings = perform_pca(df)

        # Calculate statistics (exclude Gender column if it exists)
        if 'Gender' in df.columns:
            df_numeric = df.drop('Gender', axis=1)
        else:
            df_numeric = df

        # Calculate percentiles and summary stats
        percentiles = df_numeric.quantile([0.05, 0.50, 0.95]).round(3)
        percentiles.index = ['5th Percentile', '50th Percentile', '95th Percentile']
        summary = df_numeric.describe().round(3)

        # Create Excel writer object
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Write raw data to first sheet
            df.to_excel(writer, sheet_name='Raw Data', index=False)

            # Write summary statistics to second sheet
            summary.to_excel(writer, sheet_name='Summary Statistics')

            # Write percentiles to third sheet
            percentiles.to_excel(writer, sheet_name='Percentiles')

            # Write PCA results to fourth sheet
            pca_df.to_excel(writer, sheet_name='PCA Results', index=False)

            # Write variance explained to the PCA sheet
            variance_df = pd.DataFrame({
                'Principal Component': [f'PC{i+1}' for i in range(len(variance_explained))],
                'Variance Explained (%)': [v * 100 for v in variance_explained],
                'Cumulative Variance (%)': [v * 100 for v in cumulative_variance]
            })
            variance_df.to_excel(writer, sheet_name='PCA Variance', index=False)

            # Write component loadings
            loadings.to_excel(writer, sheet_name='PCA Loadings')

    output.seek(0)

    # Set filename based on gender selection
    if gender == 'both':
        filename = 'anthropometric_analysis_combined.xlsx'
    elif gender == 'male':
        filename = 'anthropometric_analysis_male.xlsx'
    else:
        filename = 'anthropometric_analysis_female.xlsx'

    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        download_name=filename,
        as_attachment=True
    )

@app.route('/api/summary')
def api_summary():
    """API endpoint for summary statistics"""
    gender = request.args.get('gender', 'both')
    df = create_dataframe(gender)

    # Add gender information to the response
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
            if file.endswith('.png'):
                image_files.append(file)

    # Render template with available data
    return render_template('helmet_design.html', 
                          specs_exist=specs_exist,
                          image_files=image_files)

@app.route('/run_helmet_analysis', methods=['POST'])
def run_helmet_analysis():
    """Run the helmet design parameter analysis"""
    try:
        # Clear any existing image files
        for file in os.listdir('helmet_design_outputs'):
            if file.endswith('.png'):
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
@app.route('/advanced_analysis')
def advanced_analysis():
    """Advanced analysis endpoint for clustering"""
    analysis_type = request.args.get('analysis_type', '')
    gender = request.args.get('gender', 'both')
    
    if analysis_type == 'clusters':
        # Create the appropriate dataframe based on gender
        df = create_dataframe(gender)

        # Perform K-means clustering
        cluster_plot_male, clusters_male, variance_ratio_male = perform_kmeans_pca(create_dataframe_male(), n_clusters=3, gender='male') if gender in ['both', 'male'] else (None, None, None)
        cluster_plot_female, clusters_female, variance_ratio_female = perform_kmeans_pca(create_dataframe_female(), n_clusters=3, gender='female') if gender in ['both', 'female'] else (None, None, None)

        # Generate response data
        response_data = {
            'gender': gender,
            'clusters_male': clusters_male.tolist() if clusters_male is not None else None,
            'clusters_female': clusters_female.tolist() if clusters_female is not None else None,
            'cluster_plot_male': cluster_plot_male,
            'cluster_plot_female': cluster_plot_female,
            'variance_ratio_male': variance_ratio_male.tolist() if variance_ratio_male is not None else None,
            'variance_ratio_female': variance_ratio_female.tolist() if variance_ratio_female is not None else None
        }

        return jsonify(response_data)
    else:
        return jsonify({'error': 'Invalid analysis type'}), 400
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)