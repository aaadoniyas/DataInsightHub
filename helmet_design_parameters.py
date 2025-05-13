#!/usr/bin/env python3
"""
Tactical Helmet Design Parameter Analysis
Incorporating MIL-STD-1472 Human Engineering Standards
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Rectangle
import os

# Set up plotting style for military specifications
plt.style.use('dark_background')
MILITARY_GREEN = '#4A6741'
MILITARY_YELLOW = '#D4B94E'
MILITARY_RED = '#8F4539'

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
    female_means = male_means - modifications
    female_stds = male_stds * 0.9
    num_samples = 56  # Sample size for females: 56
    data = np.random.normal(loc=female_means, scale=female_stds, size=(num_samples, len(columns)))
    return pd.DataFrame(data, columns=columns)

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
    
    if "Head circumference" in available_parameters:
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
    if "Head circumference" in available_parameters:
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
    male_values = male_median.values.tolist()
    male_values += male_values[:1]  # Close the loop
    
    female_values = female_median.values.tolist()
    female_values += female_values[:1]  # Close the loop
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Normalize data for better visualization
    max_values = np.maximum(male_median, female_median)
    male_norm = male_median / max_values
    female_norm = female_median / max_values
    
    male_norm_values = male_norm.values.tolist()
    male_norm_values += male_norm_values[:1]
    
    female_norm_values = female_norm.values.tolist()
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
            ax.hlines(y=i, xmin=design_mins[i], xmax=design_max_with_clearance, 
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
        
        if "Head circumference" in available_parameters:
            size_ranges.to_excel(writer, sheet_name='Size Categories', index=False)
            size_coverage_df.to_excel(writer, sheet_name='Population Coverage', index=False)
        
        male_percentiles.to_excel(writer, sheet_name='Male Percentiles')
        female_percentiles.to_excel(writer, sheet_name='Female Percentiles')
    
    print(f"\nAnalysis complete. Results saved to {output_dir}/")
    print(f"Design specifications file: {output_dir}/tactical_helmet_specifications.xlsx")
    print(f"Visualization charts saved as PNG files in {output_dir}/")

if __name__ == "__main__":
    analyze_helmet_parameters()