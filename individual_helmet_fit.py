#!/usr/bin/env python3
"""
Individual Helmet Fit Analyzer
Checks measured anthropometric data against tactical helmet design specifications
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
matplotlib.use('Agg')

# Define styling constants
MILITARY_GREEN = '#4A6741'
MILITARY_YELLOW = '#D4B94E'
MILITARY_RED = '#8F4539'

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
        sys.exit(1)
    
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
    output_file = f"{output_dir}/individual_helmet_fit.png"
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    return output_file

def print_fit_report(result):
    """Print a formatted report of the helmet fit analysis"""
    print("\n" + "="*80)
    print("TACTICAL HELMET FIT ANALYSIS (MIL-STD-1472 Compliant)")
    print("="*80)
    
    print("\nMEASUREMENT ANALYSIS:")
    print("-"*80)
    print(f"{'Parameter':<30} {'Value (mm)':<12} {'Percentile':<12} {'Status':<15}")
    print("-"*80)
    
    for param, data in result['fit_analysis'].items():
        status_color = ''
        if data['status'] == 'WITHIN RANGE':
            status_color = '\033[92m'  # Green
        elif data['status'] == 'WITHIN CLEARANCE':
            status_color = '\033[93m'  # Yellow
        else:
            status_color = '\033[91m'  # Red
            
        percentile = f"{data['percentile']:.1f}" if isinstance(data['percentile'], (int, float)) else data['percentile']
        
        print(f"{param:<30} {data['value']:<12.1f} {percentile:<12} {status_color}{data['status']}\033[0m")
    
    print("\nSIZE RECOMMENDATION:")
    print("-"*80)
    if result['recommended_size'] is not None:
        print(f"Recommended Size: {result['recommended_size']}")
        print(f"Fit Assessment: {result['size_fit']}")
    else:
        print("Size recommendation not available. Head circumference measurement may be missing.")
    
    print("\nVISUALIZATION:")
    print("-"*80)
    print(f"Fit visualization saved to: {result['visualization_file']}")
    
    print("\nOVERALL ASSESSMENT:")
    print("-"*80)
    count_within = sum(1 for data in result['fit_analysis'].values() if data['status'] == 'WITHIN RANGE')
    count_clearance = sum(1 for data in result['fit_analysis'].values() if data['status'] == 'WITHIN CLEARANCE')
    count_outside = sum(1 for data in result['fit_analysis'].values() if data['status'] not in ['WITHIN RANGE', 'WITHIN CLEARANCE'])
    
    total = len(result['fit_analysis'])
    
    if count_outside > 0:
        print("\033[91mWARNING: Some measurements fall outside the design range.\033[0m")
        print(f"Parameters within range: {count_within}/{total}")
        print(f"Parameters within clearance: {count_clearance}/{total}")
        print(f"Parameters outside range: {count_outside}/{total}")
        
        if "Head circumference" in result['fit_analysis'] and result['fit_analysis']["Head circumference"]['status'] == 'OUTSIDE RANGE':
            print("\n\033[91mCRITICAL: Head circumference is outside the design range.\033[0m")
            print("This is the primary sizing parameter and may significantly affect helmet fit and comfort.")
    else:
        if count_clearance > 0:
            print("\033[93mNOTICE: All measurements are within design parameters, but some rely on clearance adjustments.\033[0m")
            print(f"Parameters within standard range: {count_within}/{total}")
            print(f"Parameters utilizing clearance: {count_clearance}/{total}")
        else:
            print("\033[92mEXCELLENT: All measurements are well within the standard design range.\033[0m")
    
    print("\nMIL-STD-1472 COMPLIANCE:")
    print("-"*80)
    if count_outside > 0:
        print("\033[91mNon-compliant with MIL-STD-1472 Section 4.4.1 (Population Accommodation)\033[0m")
        print("Some measurements fall outside the 5th-95th percentile design range.")
        print("Consider custom fit options for this individual.")
    else:
        print("\033[92mCompliant with MIL-STD-1472 Section 4.4.1 (Population Accommodation)\033[0m")
        print("All measurements fall within the 5th-95th percentile design range or clearance range.")


def main():
    # This is a sample run with example measurements
    # In a real application, these would be input by the user or measured
    
    # Example measurements (in mm)
    sample_measurements = {
        "Head circumference": 570,
        "Head breadth": 148,
        "Head length": 209,
        "Total head height": 229,
        "Bitragion breadth": 156,
        "Frontotemporale breadth": 141,
        "Bizygomatic breadth": 132,
        "Inion to vertex height": 181,
        "Sagittal arc": 218,
        "Bitragion arc": 226
    }
    
    # Perform fit analysis
    print("Analyzing individual measurements against tactical helmet design parameters...")
    result = check_helmet_fit(sample_measurements, gender='male')
    
    # Print report
    print_fit_report(result)
    
    print("\nNOTE: This is a sample analysis. Replace sample_measurements with actual measurements.")
    print("Example usage in interactive mode:")
    print('   result = check_helmet_fit({"Head circumference": 570, "Head breadth": 148, ...}, gender="male")')


if __name__ == "__main__":
    main()