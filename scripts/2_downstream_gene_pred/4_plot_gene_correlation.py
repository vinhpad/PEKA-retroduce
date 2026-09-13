import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse


def _results_csv(base_dir, image_encoder_name, feature_type, data_type, embedder_name):
    """Path layout written by step3_task_gene_expr_reg_KFold.py."""
    return os.path.join(
        base_dir,
        image_encoder_name,
        f"{feature_type}_gene_level_{data_type}_regression_{embedder_name}",
        "gene_regression_results.csv",
    )


def plot_gene_correlations(base_dir, output_dir, tissue,
                           image_encoder_name="H0",
                           embedder_name="scFoundation",
                           data_type="raw"):
    # Set font sizes
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 12,
        'ytick.labelsize': 14,
        'legend.fontsize': 14
    })

    # Read the three CSV files
    backbone_path = _results_csv(base_dir, image_encoder_name, "image_encoder", data_type, embedder_name)
    peka_path = _results_csv(base_dir, image_encoder_name, "peka", data_type, embedder_name)
    peka_image_path = _results_csv(base_dir, image_encoder_name, "image_encoder+peka", data_type, embedder_name)

    for path in (backbone_path, peka_path, peka_image_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing regression results: {path}")

    # Read CSV files
    backbone_df = pd.read_csv(backbone_path)
    peka_df = pd.read_csv(peka_path)
    peka_image_df = pd.read_csv(peka_image_path)

    # Sort genes based on PEKA+image results
    sorted_genes = peka_image_df.sort_values('pearson_correlation_mean')['gene'].values

    # Calculate mean PCC for each method
    backbone_mean = backbone_df['pearson_correlation_mean'].mean()
    peka_mean = peka_df['pearson_correlation_mean'].mean()
    peka_image_mean = peka_image_df['pearson_correlation_mean'].mean()

    # Create the plot with larger size for A4 paper
    plt.figure(figsize=(16, 10))

    # Plot points for each method with larger markers
    plt.scatter(range(len(sorted_genes)), 
               backbone_df.set_index('gene').loc[sorted_genes, 'pearson_correlation_mean'],
               color='gray', alpha=0.6, s=100, label=f'Backbone (mean PCC: {backbone_mean:.3f})')
    
    plt.scatter(range(len(sorted_genes)), 
               peka_df.set_index('gene').loc[sorted_genes, 'pearson_correlation_mean'],
               color='blue', alpha=0.6, s=100, label=f'PEKA (mean PCC: {peka_mean:.3f})')
    
    plt.scatter(range(len(sorted_genes)), 
               peka_image_df.set_index('gene').loc[sorted_genes, 'pearson_correlation_mean'],
               color='red', alpha=0.6, s=100, label=f'PEKA+Image (mean PCC: {peka_image_mean:.3f})')

    # Customize the plot
    plt.xticks(range(len(sorted_genes)), sorted_genes, rotation=90)
    plt.xlabel('Genes')
    plt.ylabel('Pearson Correlation Mean')
    plt.title(f'Gene-level Prediction Performance Comparison ({tissue})')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot as PDF
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'gene_correlation_comparison_{tissue}.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Plot gene-level correlation comparison')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Regression output root, i.e. the --output_root given to '
                             'step3_task_gene_expr_reg_KFold.py (e.g. OUTPUT/breast/breast_visium_26k/raw)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to write the PDF into')
    parser.add_argument('--tissue', type=str, required=True,
                        help='Tissue label used in the plot title and file name (e.g. Breast)')
    parser.add_argument('--image_encoder_name', type=str, default="H0",
                        help='Name of the image encoder model: (H0, UNI)')
    parser.add_argument('--embedder_name', type=str, default="scFoundation",
                        help='Name of the scLLM model')
    parser.add_argument('--data_type', type=str, default="raw", choices=["raw", "binned"],
                        help='Which regression variant to plot')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_gene_correlations(args.results_dir, args.output_dir, args.tissue,
                           image_encoder_name=args.image_encoder_name,
                           embedder_name=args.embedder_name,
                           data_type=args.data_type)
