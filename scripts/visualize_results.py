import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
import os

def parse_value(val):
    """Parses numeric values, handling parentheses. Returns (value, is_prismatic)."""
    if val is None:
        return np.nan, False
    
    is_prismatic = False
    if isinstance(val, str):
        # Check if the value is in parentheses
        if val.strip().startswith('(') and val.strip().endswith(')'):
            is_prismatic = True
        
        # Remove parentheses (e.g., "(0.4343)") and convert to float
        clean_val = re.sub(r'[()]\s*', '', val)
        try:
            return float(clean_val), is_prismatic
        except ValueError:
            return np.nan, False
    try:
        return float(val), is_prismatic
    except (TypeError, ValueError):
        return np.nan, False

def main():
    # Load data
    report_path = 'out/cat1_report.yml'
    if not os.path.exists(report_path):
        print(f"Error: {report_path} not found.")
        return

    print(f"Reading data from {report_path}...")
    with open(report_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    summary = data.get('summary', {})
    
    rows = []
    methods = ['aa', 'screw', 'ours']
    
    for obj_name, results in summary.items():
        for method in methods:
            m_data = results.get(method, {})
            if m_data is None:
                m_data = {}
                
            type_match = m_data.get('type_match', False)
            
            # 1. Type Match Rate
            rows.append({
                'Method': method,
                'Metric': 'Type Match Rate',
                'Value': 1 if type_match else 0,
                'Is_Prismatic': False
            })
            
            # 2. Joint Origin Error
            origin_val = np.nan
            is_prismatic = False
            if type_match:
                origin_val, is_prismatic = parse_value(m_data.get('origin_dist_m', np.nan))
            rows.append({
                'Method': method,
                'Metric': 'Joint Origin Error (meters)',
                'Value': origin_val,
                'Is_Prismatic': is_prismatic
            })
            
            # 3. Joint Axis Error
            axis_val = np.nan
            if type_match:
                axis_val, _ = parse_value(m_data.get('axis_angle_deg', np.nan))
            rows.append({
                'Method': method,
                'Metric': 'Joint Axis Error (degrees)',
                'Value': axis_val,
                'Is_Prismatic': False
            })
            
    df = pd.DataFrame(rows)
    
    # Visualization setup
    sns.set_theme(style="whitegrid")
    order = ['aa', 'screw', 'ours']
    
    plot_configs = [
        {
            'metric': 'Type Match Rate',
            'filename': 'type_match_error.png',
            'ylabel': 'Success (1) / Failure (0)',
            'title': 'Joint Type Match Rate'
        },
        {
            'metric': 'Joint Origin Error (meters)',
            'filename': 'origin_dist_error.png',
            'ylabel': 'Error (meters)',
            'title': 'Joint Origin Position Error'
        },
        {
            'metric': 'Joint Axis Error (degrees)',
            'filename': 'axis_angle_error.png',
            'ylabel': 'Error (degrees)',
            'title': 'Joint Axis Orientation Error'
        }
    ]
    
    for config in plot_configs:
        metric_name = config['metric']
        filename = config['filename']
        
        plt.figure(figsize=(10, 6))
        subset = df[df['Metric'] == metric_name].copy()
        
        # Plotting logic: only Origin Error uses prismatic split
        if metric_name == 'Joint Origin Error (meters)':
            subset_normal = subset[subset['Is_Prismatic'] == False]
            subset_prismatic = subset[subset['Is_Prismatic'] == True]
            
            # 1. Plot normal (revolute) dots — solid filled
            if not subset_normal.empty:
                sns.stripplot(
                    data=subset_normal, 
                    x='Method', y='Value', order=order, 
                    size=12, jitter=0.2, alpha=0.8, 
                    palette="Set2", hue='Method', legend=False
                )
            
            # 2. Plot prismatic dots — transparent fill, colored border only
            # seaborn ignores facecolors='none', so we use matplotlib scatter directly
            if not subset_prismatic.empty:
                ax = plt.gca()
                # Set2 palette colors matching order
                set2_colors = sns.color_palette("Set2", len(order))
                method_color = {m: set2_colors[i] for i, m in enumerate(order)}
                rng = np.random.default_rng(seed=42)  # fixed seed for reproducibility
                for i, method in enumerate(order):
                    pts = subset_prismatic[subset_prismatic['Method'] == method]['Value'].dropna()
                    if not pts.empty:
                        jitter_offsets = rng.uniform(-0.2, 0.2, size=len(pts))
                        ax.scatter(
                            x=i + jitter_offsets,
                            y=pts.values,
                            s=12**2,  # size=12 in stripplot → s=size^2 for scatter
                            facecolors='none',
                            edgecolors=method_color[method],
                            linewidths=2,
                            zorder=3
                        )
            
            # Mean uses ONLY non-prismatic (revolute) points
            medians = subset_normal.groupby('Method', observed=True)['Value'].mean().reindex(order)
        else:
            # All other metrics: plot all points, include all in median
            sns.stripplot(
                data=subset, 
                x='Method', y='Value', order=order, 
                size=12, jitter=0.2, alpha=0.8, 
                palette="Set2", hue='Method', legend=False
            )
            medians = subset.groupby('Method', observed=True)['Value'].mean().reindex(order)
        
        # Add a short horizontal line for each median
        for i, method in enumerate(order):
            m_val = medians.get(method, np.nan)
            if not np.isnan(m_val):
                plt.plot([i - 0.25, i + 0.25], [m_val, m_val], color='#e74c3c', lw=5, solid_capstyle='round', zorder=5)
                plt.text(i, m_val, f'{m_val:.3f}', color='black', ha='center', va='bottom', fontsize=10, fontweight='bold', zorder=6)
        
        plt.title(config['title'], fontsize=16, fontweight='bold', pad=20)
        plt.ylabel(config['ylabel'], fontsize=13)
        plt.xlabel("Method", fontsize=13)
        
        # Special handling for Type Match Rate scale
        if metric_name == 'Type Match Rate':
            plt.ylim(-0.2, 1.2)
            plt.yticks([0, 1], ['Failure (0)', 'Success (1)'])
            
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Created and saved: {filename}")

if __name__ == "__main__":
    main()
