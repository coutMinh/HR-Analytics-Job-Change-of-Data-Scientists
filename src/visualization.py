"""
Visualization Functions for HR Analytics (NumPy + Matplotlib + Seaborn)
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===== HELPER FUNCTIONS =====
def _setup_plot(nrows=1, ncols=1, figsize=None):
    """Create figure with consistent styling"""
    if figsize is None:
        figsize = (14, 5) if ncols > 1 else (10, 6)
    return plt.subplots(nrows, ncols, figsize=figsize)

def _add_bar_labels(ax, values, is_percentage=False):
    """Add value labels on bars"""
    for i, v in enumerate(values):
        label = f'{v:.1f}%' if is_percentage else f'{v:,.0f}'
        ax.text(i, v + (1 if is_percentage else 500), label, ha='center', fontsize=10)

# ===== BASIC VISUALIZATIONS =====
def plot_target_distribution(target_counts, target_pct):
    """Target variable distribution (count + pie)"""
    fig, (ax1, ax2) = _setup_plot(1, 2, (14, 5))
    
    labels = ['Not Looking (0)', 'Looking (1)']
    values = [target_counts[0], target_counts[1]]
    colors = ['#2ecc71', '#e74c3c']
    
    # Count bar chart
    ax1.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Target Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    _add_bar_labels(ax1, values)
    
    # Pie chart
    ax2.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, 
            startangle=90, explode=(0.05, 0.05), shadow=True)
    ax2.set_title('Target %', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_categorical_vs_target(cat_stats, title, sort_by='percentage'):
    """Categorical feature vs target (grouped bar + rate)"""
    fig, (ax1, ax2) = _setup_plot(1, 2, (14, 5))
    
    cats = list(cat_stats.keys())
    not_looking = [cat_stats[c]['not_looking'] for c in cats]
    looking = [cat_stats[c]['looking'] for c in cats]
    pcts = [cat_stats[c]['percentage'] for c in cats]
    
    x = np.arange(len(cats))
    width = 0.35
    
    # Grouped bars
    ax1.bar(x - width/2, not_looking, width, label='Not Looking', color='#3498db', alpha=0.8)
    ax1.bar(x + width/2, looking, width, label='Looking', color='#e74c3c', alpha=0.8)
    ax1.set_xlabel(title, fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title(f'Job Change by {title}', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(cats, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Percentage bars
    ax2.bar(cats, pcts, color='#e67e22', alpha=0.8, edgecolor='black')
    ax2.set_xlabel(title, fontsize=12)
    ax2.set_ylabel('Job Change Rate (%)', fontsize=12)
    ax2.set_title(f'Rate by {title}', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(cats, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    _add_bar_labels(ax2, pcts, is_percentage=True)
    
    plt.tight_layout()
    plt.show()

def plot_numeric_vs_target(numeric_valid, target_valid, feature_name, bins=30):
    """Numeric feature analysis (distribution + boxplot)"""
    fig, axes = _setup_plot(2, 2, (16, 12))
    
    data_0 = numeric_valid[target_valid == 0]
    data_1 = numeric_valid[target_valid == 1]
    
    # Overall distribution
    axes[0,0].hist(numeric_valid, bins=bins, color='teal', alpha=0.7, edgecolor='black')
    axes[0,0].set_xlabel(feature_name, fontsize=12)
    axes[0,0].set_ylabel('Frequency', fontsize=12)
    axes[0,0].set_title(f'{feature_name} Distribution', fontsize=14, fontweight='bold')
    axes[0,0].grid(axis='y', alpha=0.3)
    
    # By target
    axes[0,1].hist([data_0, data_1], bins=bins, label=['Not Looking', 'Looking'],
                   color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
    axes[0,1].set_xlabel(feature_name, fontsize=12)
    axes[0,1].set_ylabel('Frequency', fontsize=12)
    axes[0,1].set_title(f'{feature_name} by Target', fontsize=14, fontweight='bold')
    axes[0,1].legend()
    axes[0,1].grid(axis='y', alpha=0.3)
    
    # Box plot
    bp = axes[1,0].boxplot([data_0, data_1], labels=['Not Looking', 'Looking'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#3498db', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1,0].set_ylabel(feature_name, fontsize=12)
    axes[1,0].set_title(f'{feature_name} Box Plot', fontsize=14, fontweight='bold')
    axes[1,0].grid(axis='y', alpha=0.3)
    
    return fig, axes, data_0, data_1

def plot_binned_analysis(numeric_valid, target_valid, feature_name, bin_edges, bin_labels):
    """Job change rate by binned numeric feature"""
    n_bins = len(bin_labels)
    binned = np.digitize(numeric_valid, bin_edges[:-1]) - 1
    binned = np.clip(binned, 0, n_bins - 1)
    
    rates = []
    for i in range(n_bins):
        mask = binned == i
        rate = np.sum(target_valid[mask] == 1) / np.sum(mask) * 100 if np.sum(mask) > 0 else 0
        rates.append(rate)
    
    fig, ax = _setup_plot()
    ax.bar(range(n_bins), rates, color='orange', alpha=0.8, edgecolor='black')
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax.set_xlabel(f'{feature_name} Range', fontsize=12)
    ax.set_ylabel('Job Change Rate (%)', fontsize=12)
    ax.set_title(f'Rate by {feature_name}', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    _add_bar_labels(ax, rates, is_percentage=True)
    plt.tight_layout()
    plt.show()

# ===== ADVANCED VISUALIZATIONS =====
def plot_education_analysis(edu_stats):
    """Comprehensive education analysis (4 subplots)"""
    fig, axes = _setup_plot(2, 2, (16, 12))
    
    edu_names = list(edu_stats.keys())
    counts = [edu_stats[e]['count'] for e in edu_names]
    pcts = [edu_stats[e]['percentage'] for e in edu_names]
    not_looking = [edu_stats[e]['not_looking'] for e in edu_names]
    looking = [edu_stats[e]['looking'] for e in edu_names]
    
    # Distribution
    axes[0,0].barh(edu_names, counts, color='skyblue', edgecolor='black')
    axes[0,0].set_xlabel('Count', fontsize=12)
    axes[0,0].set_title('Education Distribution', fontsize=14, fontweight='bold')
    axes[0,0].grid(axis='x', alpha=0.3)
    
    # Job change rate
    sorted_data = sorted(zip(edu_names, pcts), key=lambda x: x[1])
    axes[0,1].barh([x[0] for x in sorted_data], [x[1] for x in sorted_data], 
                   color='coral', edgecolor='black')
    axes[0,1].set_xlabel('Job Change Rate (%)', fontsize=12)
    axes[0,1].set_title('Rate by Education', fontsize=14, fontweight='bold')
    axes[0,1].grid(axis='x', alpha=0.3)
    
    # Grouped bars
    x = np.arange(len(edu_names))
    axes[1,0].bar(x - 0.2, not_looking, 0.4, label='Not Looking', color='#3498db', alpha=0.8)
    axes[1,0].bar(x + 0.2, looking, 0.4, label='Looking', color='#e74c3c', alpha=0.8)
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(edu_names, rotation=45, ha='right')
    axes[1,0].set_ylabel('Count', fontsize=12)
    axes[1,0].set_title('Counts by Education', fontsize=14, fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(axis='y', alpha=0.3)
    
    # Stacked %
    not_looking_pct = [edu_stats[e]['not_looking']/edu_stats[e]['count']*100 for e in edu_names]
    looking_pct = [edu_stats[e]['looking']/edu_stats[e]['count']*100 for e in edu_names]
    axes[1,1].bar(x, not_looking_pct, label='Not Looking', color='#3498db', alpha=0.8)
    axes[1,1].bar(x, looking_pct, bottom=not_looking_pct, label='Looking', color='#e74c3c', alpha=0.8)
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(edu_names, rotation=45, ha='right')
    axes[1,1].set_ylabel('Percentage', fontsize=12)
    axes[1,1].set_title('Stacked %', fontsize=14, fontweight='bold')
    axes[1,1].legend()
    axes[1,1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_experience_analysis(exp_sorted):
    """Experience analysis (4 subplots)"""
    fig, axes = _setup_plot(2, 2, (16, 12))
    
    exp_names = [x[0] for x in exp_sorted]
    counts = [x[1]['count'] for x in exp_sorted]
    pcts = [x[1]['percentage'] for x in exp_sorted]
    not_looking = [x[1]['not_looking'] for x in exp_sorted]
    looking = [x[1]['looking'] for x in exp_sorted]
    
    # Distribution
    axes[0,0].bar(range(len(exp_names)), counts, color='lightgreen', edgecolor='black')
    axes[0,0].set_xticks(range(len(exp_names)))
    axes[0,0].set_xticklabels(exp_names, rotation=45)
    axes[0,0].set_ylabel('Count', fontsize=12)
    axes[0,0].set_title('Experience Distribution', fontsize=14, fontweight='bold')
    axes[0,0].grid(axis='y', alpha=0.3)
    
    # Rate line plot
    axes[0,1].plot(range(len(exp_names)), pcts, marker='o', linewidth=2, markersize=8, color='#e74c3c')
    axes[0,1].fill_between(range(len(exp_names)), pcts, alpha=0.3, color='#e74c3c')
    axes[0,1].set_xticks(range(len(exp_names)))
    axes[0,1].set_xticklabels(exp_names, rotation=45)
    axes[0,1].set_ylabel('Job Change Rate (%)', fontsize=12)
    axes[0,1].set_title('Rate by Experience', fontsize=14, fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)
    
    # Grouped bars
    x = np.arange(len(exp_names))
    axes[1,0].bar(x - 0.2, not_looking, 0.4, label='Not Looking', color='#3498db', alpha=0.8)
    axes[1,0].bar(x + 0.2, looking, 0.4, label='Looking', color='#e74c3c', alpha=0.8)
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(exp_names, rotation=45)
    axes[1,0].set_ylabel('Count', fontsize=12)
    axes[1,0].set_title('Counts by Experience', fontsize=14, fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(axis='y', alpha=0.3)
    
    # Rate bars
    axes[1,1].bar(range(len(exp_names)), pcts, color='green', alpha=0.8, edgecolor='black')
    axes[1,1].set_xticks(range(len(exp_names)))
    axes[1,1].set_xticklabels(exp_names, rotation=45)
    axes[1,1].set_ylabel('Job Change Rate (%)', fontsize=12)
    axes[1,1].set_title('Rate Distribution', fontsize=14, fontweight='bold')
    axes[1,1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_company_analysis(size_stats, type_stats):
    """Company size & type analysis"""
    fig, axes = _setup_plot(2, 2, (16, 12))
    
    # Size distribution
    size_names = list(size_stats.keys())
    size_counts = [size_stats[s]['count'] for s in size_names]
    axes[0,0].barh(size_names, size_counts, color='lightcoral', edgecolor='black')
    axes[0,0].set_xlabel('Count', fontsize=12)
    axes[0,0].set_title('Company Size Distribution', fontsize=14, fontweight='bold')
    axes[0,0].grid(axis='x', alpha=0.3)
    
    # Size rate
    size_sorted = sorted([(s, size_stats[s]['percentage']) for s in size_names], key=lambda x: x[1])
    axes[0,1].barh([x[0] for x in size_sorted], [x[1] for x in size_sorted], 
                   color='steelblue', edgecolor='black')
    axes[0,1].set_xlabel('Job Change Rate (%)', fontsize=12)
    axes[0,1].set_title('Rate by Size', fontsize=14, fontweight='bold')
    axes[0,1].grid(axis='x', alpha=0.3)
    
    # Type distribution
    type_names = list(type_stats.keys())
    type_counts = [type_stats[t]['count'] for t in type_names]
    axes[1,0].bar(range(len(type_names)), type_counts, color='lightgreen', alpha=0.8, edgecolor='black')
    axes[1,0].set_xticks(range(len(type_names)))
    axes[1,0].set_xticklabels(type_names, rotation=45, ha='right')
    axes[1,0].set_ylabel('Count', fontsize=12)
    axes[1,0].set_title('Company Type Distribution', fontsize=14, fontweight='bold')
    axes[1,0].grid(axis='y', alpha=0.3)
    
    # Type rate
    type_sorted = sorted([(t, type_stats[t]['percentage']) for t in type_names], key=lambda x: x[1], reverse=True)
    axes[1,1].bar(range(len(type_sorted)), [x[1] for x in type_sorted], color='coral', alpha=0.8, edgecolor='black')
    axes[1,1].set_xticks(range(len(type_sorted)))
    axes[1,1].set_xticklabels([x[0] for x in type_sorted], rotation=45, ha='right')
    axes[1,1].set_ylabel('Job Change Rate (%)', fontsize=12)
    axes[1,1].set_title('Rate by Type', fontsize=14, fontweight='bold')
    axes[1,1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_university_analysis(univ_stats):
    """University enrollment analysis"""
    fig, axes = _setup_plot(1, 3, (18, 5))
    
    names = list(univ_stats.keys())
    counts = [univ_stats[u]['count'] for u in names]
    pcts = [univ_stats[u]['percentage'] for u in names]
    
    # Pie chart
    axes[0].pie(counts, labels=names, autopct='%1.1f%%', startangle=90, explode=[0.05]*len(names))
    axes[0].set_title('University Enrollment', fontsize=14, fontweight='bold')
    
    # Grouped bars
    x = np.arange(len(names))
    not_looking = [univ_stats[u]['not_looking'] for u in names]
    looking = [univ_stats[u]['looking'] for u in names]
    axes[1].bar(x - 0.2, not_looking, 0.4, label='Not Looking', color='#3498db', alpha=0.8)
    axes[1].bar(x + 0.2, looking, 0.4, label='Looking', color='#e74c3c', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha='right')
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Job Change by Enrollment', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    # Rates
    sorted_data = sorted(zip(names, pcts), key=lambda x: x[1], reverse=True)
    axes[2].bar(range(len(names)), [x[1] for x in sorted_data], color='darkviolet', alpha=0.8, edgecolor='black')
    axes[2].set_xticks(range(len(names)))
    axes[2].set_xticklabels([x[0] for x in sorted_data], rotation=45, ha='right')
    axes[2].set_ylabel('Job Change Rate (%)', fontsize=12)
    axes[2].set_title('Rate by Enrollment', fontsize=14, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_gender_analysis(gender_stats):
    """Gender analysis"""
    fig, axes = _setup_plot(1, 3, (16, 5))
    
    names = list(gender_stats.keys())
    counts = [gender_stats[g]['count'] for g in names]
    pcts = [gender_stats[g]['percentage'] for g in names]
    colors = ['steelblue', 'pink', 'lightgreen'][:len(names)]
    
    # Distribution
    axes[0].bar(names, counts, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Gender Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Grouped
    x = np.arange(len(names))
    not_looking = [gender_stats[g]['not_looking'] for g in names]
    looking = [gender_stats[g]['looking'] for g in names]
    axes[1].bar(x - 0.2, not_looking, 0.4, label='Not Looking', color='#3498db', alpha=0.8)
    axes[1].bar(x + 0.2, looking, 0.4, label='Looking', color='#e74c3c', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Job Change by Gender', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    # Rates
    axes[2].bar(names, pcts, color=colors, alpha=0.8, edgecolor='black')
    axes[2].set_ylabel('Job Change Rate (%)', fontsize=12)
    axes[2].set_title('Rate by Gender', fontsize=14, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)
    _add_bar_labels(axes[2], pcts, is_percentage=True)
    
    plt.tight_layout()
    plt.show()

def plot_last_job_analysis(last_job_sorted):
    """Last job change analysis"""
    fig, axes = _setup_plot(2, 2, (16, 12))
    
    names = [x[0] for x in last_job_sorted]
    counts = [x[1]['count'] for x in last_job_sorted]
    pcts = [x[1]['percentage'] for x in last_job_sorted]
    
    # Distribution
    axes[0,0].bar(range(len(names)), counts, color='orchid', alpha=0.8, edgecolor='black')
    axes[0,0].set_xticks(range(len(names)))
    axes[0,0].set_xticklabels(names, rotation=45)
    axes[0,0].set_ylabel('Count', fontsize=12)
    axes[0,0].set_title('Years Since Last Job', fontsize=14, fontweight='bold')
    axes[0,0].grid(axis='y', alpha=0.3)
    
    # Rate line
    axes[0,1].plot(range(len(names)), pcts, marker='o', linewidth=2, markersize=10, color='crimson')
    axes[0,1].fill_between(range(len(names)), pcts, alpha=0.3, color='crimson')
    axes[0,1].set_xticks(range(len(names)))
    axes[0,1].set_xticklabels(names, rotation=45)
    axes[0,1].set_ylabel('Job Change Rate (%)', fontsize=12)
    axes[0,1].set_title('Rate by Last Job', fontsize=14, fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)
    
    # Grouped
    x = np.arange(len(names))
    not_looking = [x[1]['not_looking'] for x in last_job_sorted]
    looking = [x[1]['looking'] for x in last_job_sorted]
    axes[1,0].bar(x - 0.2, not_looking, 0.4, label='Not Looking', color='#3498db', alpha=0.8)
    axes[1,0].bar(x + 0.2, looking, 0.4, label='Looking', color='#e74c3c', alpha=0.8)
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(names, rotation=45)
    axes[1,0].set_ylabel('Count', fontsize=12)
    axes[1,0].set_title('Counts by Last Job', fontsize=14, fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(axis='y', alpha=0.3)
    
    # Stacked
    not_looking_pct = [x[1]['not_looking']/x[1]['count']*100 for x in last_job_sorted]
    looking_pct = [x[1]['looking']/x[1]['count']*100 for x in last_job_sorted]
    axes[1,1].bar(x, not_looking_pct, label='Not Looking', color='#3498db', alpha=0.8)
    axes[1,1].bar(x, looking_pct, bottom=not_looking_pct, label='Looking', color='#e74c3c', alpha=0.8)
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(names, rotation=45)
    axes[1,1].set_ylabel('Percentage', fontsize=12)
    axes[1,1].set_title('Stacked %', fontsize=14, fontweight='bold')
    axes[1,1].legend()
    axes[1,1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_correlation_analysis(corr_names, target_correlations, corr_matrix, target_idx):
    """Correlation analysis (bar + heatmap)"""
    fig, (ax1, ax2) = _setup_plot(1, 2, (16, 6))
    
    # Correlation bars
    corr_data = [(corr_names[i].replace('_enc', ''), target_correlations[i]) 
                 for i in range(len(corr_names)) if corr_names[i] != 'target']
    corr_sorted = sorted(corr_data, key=lambda x: x[1])
    names, values = [x[0] for x in corr_sorted], [x[1] for x in corr_sorted]
    colors = ['green' if x > 0 else 'red' for x in values]
    
    ax1.barh(range(len(names)), values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names)
    ax1.set_xlabel('Correlation with Target', fontsize=12)
    ax1.set_title('Feature Correlation', fontsize=14, fontweight='bold')
    ax1.axvline(x=0, color='black', linewidth=0.8)
    ax1.grid(axis='x', alpha=0.3)
    
    # Heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, ax=ax2,
                xticklabels=[n.replace('_enc', '')[:15] for n in corr_names],
                yticklabels=[n.replace('_enc', '')[:15] for n in corr_names],
                cbar_kws={'label': 'Correlation'})
    ax2.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

