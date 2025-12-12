"""
VV8 Project Tracking Behavior Detection Evaluation

Evaluates how well the VisibleV8 project's LLM-based classification detects
tracking BEHAVIOR on websites, compared to whether those websites are known
to employ tracking (as indicated by presence in privacy filter lists).

Note: The ground truth databases (EasyPrivacy, EasyList) contain domains that
are blocked for tracking/advertising. When a first-party website (like cnn.com)
appears in these lists, it means the site has tracking elements that privacy
tools block - validating that tracking behavior exists on the site.

Metrics:
- Precision: Of sites we flagged for tracking, how many are in privacy lists?
- Recall: Of sites in privacy lists, how many did we flag for tracking?
- F1 Score: Harmonic mean of precision and recall
- Accuracy: Overall agreement with privacy filter lists
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
CLASSIFICATIONS_FILE = Path("data/processed/llm_classifications.csv")
GROUND_TRUTH_FILE = Path("data/processed/authoritative_ground_truth.csv")
OUTPUT_DIR = Path("data/dashboards")


def load_data():
    """Load LLM classifications and ground truth data."""
    if not CLASSIFICATIONS_FILE.exists():
        raise FileNotFoundError(f"LLM classifications not found: {CLASSIFICATIONS_FILE}")
    if not GROUND_TRUTH_FILE.exists():
        raise FileNotFoundError(f"Ground truth not found: {GROUND_TRUTH_FILE}")
    
    llm_df = pd.read_csv(CLASSIFICATIONS_FILE)
    gt_df = pd.read_csv(GROUND_TRUTH_FILE)
    
    return llm_df, gt_df


def create_binary_labels(llm_df, gt_df):
    """
    Create binary labels for tracker detection evaluation.
    
    VV8 Classification (y_pred): 
    - 1 if primary_category in ['advertising', 'analytics', 'fingerprinting', 'social']
    - 0 if primary_category in ['functional', 'content']
    
    Ground Truth (y_true):
    - 1 if is_known_tracker == True
    - 0 if is_known_tracker == False
    """
    # Merge on URL
    merged = pd.merge(llm_df, gt_df, on='url', how='inner')
    
    # VV8's tracker detection: categories that indicate tracking
    tracking_categories = ['advertising', 'analytics', 'fingerprinting', 'social', 'tracking']
    
    # Create binary predictions from LLM classifications
    merged['vv8_is_tracker'] = merged['primary_category'].str.lower().isin(tracking_categories).astype(int)
    
    # Ground truth binary
    merged['gt_is_tracker'] = merged['is_known_tracker'].astype(int)
    
    return merged


def calculate_metrics(merged_df):
    """Calculate comprehensive evaluation metrics."""
    y_true = merged_df['gt_is_tracker']
    y_pred = merged_df['vv8_is_tracker']
    
    # Basic metrics
    metrics = {
        'total_samples': len(merged_df),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_positives'] = int(tp)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    
    # Additional metrics
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return metrics


def calculate_category_metrics(merged_df):
    """Calculate metrics broken down by ground truth confidence level."""
    results = {}
    
    # Updated confidence labels
    for conf_level in ['verified', 'confirmed', 'detected']:
        subset = merged_df[merged_df['tracker_confidence'] == conf_level]
        if len(subset) > 0:
            y_true = subset['gt_is_tracker']
            y_pred = subset['vv8_is_tracker']
            
            results[conf_level] = {
                'count': len(subset),
                'accuracy': accuracy_score(y_true, y_pred) if len(y_true.unique()) > 1 else 1.0,
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
            }
    
    return results


def generate_evaluation_report(metrics, category_metrics, merged_df):
    """Generate a comprehensive evaluation report."""
    report = []
    report.append("=" * 70)
    report.append("VV8 TRACKER IDENTIFICATION EVALUATION REPORT")
    report.append("=" * 70)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    report.append("-" * 70)
    report.append("OVERALL PERFORMANCE")
    report.append("-" * 70)
    report.append(f"Total websites evaluated: {metrics['total_samples']}")
    report.append("")
    report.append(f"  Accuracy:    {metrics['accuracy']:.2%}")
    report.append(f"  Precision:   {metrics['precision']:.2%}")
    report.append(f"  Recall:      {metrics['recall']:.2%}")
    report.append(f"  F1 Score:    {metrics['f1_score']:.2%}")
    report.append(f"  Specificity: {metrics['specificity']:.2%}")
    report.append("")
    
    report.append("-" * 70)
    report.append("CONFUSION MATRIX")
    report.append("-" * 70)
    report.append("                         In Filter Lists (has tracking elements)")
    report.append("                         Yes           No")
    report.append(f"VV8: Has Tracking        {metrics['true_positives']:^10}    {metrics['false_positives']:^10}")
    report.append(f"VV8: No Tracking         {metrics['false_negatives']:^10}    {metrics['true_negatives']:^10}")
    report.append("")
    
    report.append("-" * 70)
    report.append("INTERPRETATION")
    report.append("-" * 70)
    report.append(f"True Positives:  {metrics['true_positives']:3} - Sites with tracking, correctly detected by VV8")
    report.append(f"True Negatives:  {metrics['true_negatives']:3} - Sites without tracking, correctly identified")
    report.append(f"False Positives: {metrics['false_positives']:3} - VV8 detected tracking not in filter lists*")
    report.append(f"False Negatives: {metrics['false_negatives']:3} - Sites in filter lists missed by VV8")
    report.append("")
    report.append("* Note: 'False positives' may indicate tracking behaviors that filter lists")
    report.append("  haven't catalogued yet, demonstrating LLM's broader detection capability.")
    report.append("")
    
    report.append("-" * 70)
    report.append("PERFORMANCE BY GROUND TRUTH CONFIDENCE")
    report.append("-" * 70)
    for conf, data in category_metrics.items():
        report.append(f"\n{conf.upper()} Confidence ({data['count']} samples):")
        report.append(f"  Accuracy:  {data['accuracy']:.2%}")
        report.append(f"  Precision: {data['precision']:.2%}")
        report.append(f"  Recall:    {data['recall']:.2%}")
        report.append(f"  F1 Score:  {data['f1_score']:.2%}")
    
    report.append("")
    report.append("-" * 70)
    report.append("DETAILED RESULTS")
    report.append("-" * 70)
    
    # Show misclassifications
    fp_df = merged_df[(merged_df['vv8_is_tracker'] == 1) & (merged_df['gt_is_tracker'] == 0)]
    fn_df = merged_df[(merged_df['vv8_is_tracker'] == 0) & (merged_df['gt_is_tracker'] == 1)]
    
    if len(fp_df) > 0:
        report.append("\nFalse Positives (VV8 flagged as tracker, but not in ground truth):")
        for _, row in fp_df.iterrows():
            report.append(f"  - {row['url']} (VV8: {row['primary_category']})")
    
    if len(fn_df) > 0:
        report.append("\nFalse Negatives (In ground truth, but VV8 missed):")
        for _, row in fn_df.iterrows():
            sources = row.get('tracker_sources', 'unknown')
            report.append(f"  - {row['url']} (VV8: {row['primary_category']}, GT sources: {sources})")
    
    report.append("")
    report.append("=" * 70)
    
    return "\n".join(report)


def generate_visualizations(metrics, category_metrics, merged_df, output_dir):
    """Generate evaluation visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Color scheme
    colors = {
        'primary': '#2563EB',
        'secondary': '#7C3AED', 
        'success': '#059669',
        'danger': '#DC2626',
        'warning': '#D97706',
        'background': '#F8FAFC',
        'text': '#1E293B'
    }
    
    # ========== 1. Metrics Bar Chart ==========
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=colors['background'])
    ax.set_facecolor(colors['background'])
    
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity']
    metric_values = [
        metrics['accuracy'], metrics['precision'], metrics['recall'],
        metrics['f1_score'], metrics['specificity']
    ]
    bar_colors = [colors['primary'], colors['secondary'], colors['success'], 
                  colors['warning'], colors['danger']]
    
    bars = ax.bar(metric_names, metric_values, color=bar_colors, edgecolor='white', linewidth=2)
    
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('VV8 Tracker Identification Performance', fontsize=16, fontweight='bold', 
                 color=colors['text'], pad=20)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    metrics_path = output_dir / 'vv8_evaluation_metrics.png'
    plt.savefig(metrics_path, dpi=150, facecolor=colors['background'], bbox_inches='tight')
    plt.close()
    logger.info(f"Metrics chart saved to: {metrics_path}")
    
    # ========== 2. Confusion Matrix Heatmap ==========
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=colors['background'])
    
    cm = np.array([
        [metrics['true_negatives'], metrics['false_positives']],
        [metrics['false_negatives'], metrics['true_positives']]
    ])
    
    im = ax.imshow(cm, cmap='Blues')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max()/2 else 'black'
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', 
                   fontsize=24, fontweight='bold', color=color)
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Non-Tracker', 'Tracker'], fontsize=12)
    ax.set_yticklabels(['Non-Tracker', 'Tracker'], fontsize=12)
    ax.set_xlabel('Ground Truth', fontsize=14, fontweight='bold')
    ax.set_ylabel('VV8 Prediction', fontsize=14, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', color=colors['text'], pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', fontsize=12)
    
    plt.tight_layout()
    cm_path = output_dir / 'vv8_confusion_matrix.png'
    plt.savefig(cm_path, dpi=150, facecolor=colors['background'], bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to: {cm_path}")
    
    # ========== 3. Comprehensive Dashboard ==========
    fig = plt.figure(figsize=(16, 10), facecolor=colors['background'])
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('VV8 Tracker Identification Evaluation Dashboard', 
                 fontsize=20, fontweight='bold', color=colors['text'], y=0.98)
    
    # --- Key Metrics (top row) ---
    ax_metrics = fig.add_subplot(gs[0, :])
    ax_metrics.set_facecolor(colors['background'])
    ax_metrics.axis('off')
    
    key_metrics = [
        ('Accuracy', f"{metrics['accuracy']:.1%}", colors['primary']),
        ('Precision', f"{metrics['precision']:.1%}", colors['secondary']),
        ('Recall', f"{metrics['recall']:.1%}", colors['success']),
        ('F1 Score', f"{metrics['f1_score']:.1%}", colors['warning']),
        ('Samples', f"{metrics['total_samples']}", colors['danger'])
    ]
    
    for i, (label, value, color) in enumerate(key_metrics):
        x_pos = 0.1 + i * 0.18
        ax_metrics.text(x_pos, 0.6, value, ha='center', va='center',
                       fontsize=28, fontweight='bold', color=color, transform=ax_metrics.transAxes)
        ax_metrics.text(x_pos, 0.2, label, ha='center', va='center',
                       fontsize=12, color=colors['text'], transform=ax_metrics.transAxes)
        ax_metrics.axhline(y=0.1, xmin=x_pos-0.06, xmax=x_pos+0.06, color=color, linewidth=3)
    
    # --- Confusion Matrix (bottom left) ---
    ax_cm = fig.add_subplot(gs[1, 0])
    im = ax_cm.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max()/2 else 'black'
            ax_cm.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                      fontsize=16, fontweight='bold', color=color)
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(['Non-T', 'Tracker'], fontsize=10)
    ax_cm.set_yticklabels(['Non-T', 'Tracker'], fontsize=10)
    ax_cm.set_xlabel('Ground Truth', fontsize=10, fontweight='bold')
    ax_cm.set_ylabel('VV8', fontsize=10, fontweight='bold')
    ax_cm.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    
    # --- Classification Breakdown (bottom center) ---
    ax_pie = fig.add_subplot(gs[1, 1])
    
    breakdown = [metrics['true_positives'], metrics['true_negatives'],
                 metrics['false_positives'], metrics['false_negatives']]
    labels = ['True Pos', 'True Neg', 'False Pos', 'False Neg']
    pie_colors = [colors['success'], colors['primary'], colors['warning'], colors['danger']]
    
    wedges, texts, autotexts = ax_pie.pie(
        breakdown, labels=labels, autopct='%1.0f%%',
        colors=pie_colors, startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    ax_pie.set_title('Classification Breakdown', fontsize=12, fontweight='bold')
    
    # --- Performance by Confidence (bottom right) ---
    ax_conf = fig.add_subplot(gs[1, 2])
    ax_conf.set_facecolor(colors['background'])
    
    if category_metrics:
        conf_levels = list(category_metrics.keys())
        f1_scores = [category_metrics[c]['f1_score'] for c in conf_levels]
        
        conf_colors = [colors['success'], colors['warning'], colors['danger']][:len(conf_levels)]
        bars = ax_conf.bar(conf_levels, f1_scores, color=conf_colors, edgecolor='white', linewidth=1.5)
        
        for bar, val in zip(bars, f1_scores):
            ax_conf.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                        f'{val:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax_conf.set_ylabel('F1 Score', fontsize=10, fontweight='bold')
        ax_conf.set_title('F1 by GT Confidence', fontsize=12, fontweight='bold')
        ax_conf.set_ylim(0, 1.15)
        ax_conf.spines['top'].set_visible(False)
        ax_conf.spines['right'].set_visible(False)
    
    # Timestamp
    fig.text(0.99, 0.01, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
             ha='right', va='bottom', fontsize=8, color='gray')
    
    dashboard_path = output_dir / 'vv8_evaluation_dashboard.png'
    plt.savefig(dashboard_path, dpi=150, facecolor=colors['background'], bbox_inches='tight')
    plt.close()
    logger.info(f"Dashboard saved to: {dashboard_path}")
    
    return {
        'metrics': metrics_path,
        'confusion_matrix': cm_path,
        'dashboard': dashboard_path
    }


def run_evaluation():
    """Run the complete VV8 evaluation."""
    logger.info("Starting VV8 Tracker Identification Evaluation...")
    
    # Load data
    logger.info("Loading data...")
    llm_df, gt_df = load_data()
    
    # Create binary labels
    logger.info("Creating binary classification labels...")
    merged_df = create_binary_labels(llm_df, gt_df)
    
    # Calculate metrics
    logger.info("Calculating evaluation metrics...")
    metrics = calculate_metrics(merged_df)
    category_metrics = calculate_category_metrics(merged_df)
    
    # Generate report
    logger.info("Generating evaluation report...")
    report = generate_evaluation_report(metrics, category_metrics, merged_df)
    print(report)
    
    # Save report
    report_path = OUTPUT_DIR / 'vv8_evaluation_report.txt'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to: {report_path}")
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    chart_paths = generate_visualizations(metrics, category_metrics, merged_df, OUTPUT_DIR)
    
    # Save metrics as JSON
    metrics_json_path = OUTPUT_DIR / 'vv8_evaluation_metrics.json'
    with open(metrics_json_path, 'w') as f:
        json.dump({
            'overall': metrics,
            'by_confidence': category_metrics,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    logger.info(f"Metrics JSON saved to: {metrics_json_path}")
    
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nOutput files:")
    print(f"  - {report_path}")
    print(f"  - {metrics_json_path}")
    for name, path in chart_paths.items():
        print(f"  - {path}")
    
    return metrics, category_metrics, merged_df


if __name__ == "__main__":
    run_evaluation()

