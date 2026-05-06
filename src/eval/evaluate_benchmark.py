"""
Evaluation script for Closed-Form Matting on alphamatting.com dataset.
Computes SAD, MSE, MAD metrics and generates visualizations.
"""

import numpy as np
import cv2
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from closed_form_matting import (
    solve_closed_form_matting,
    composite_on_background,
    compute_matting_metrics
)


class MattingEvaluator:
    """Evaluator for matting algorithms."""

    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.viz_dir = self.output_dir / "visualizations"
        self.alphas_dir = self.output_dir / "predicted_alphas"

        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.alphas_dir.mkdir(parents=True, exist_ok=True)

        # Dataset paths
        self.images_dir = self.data_dir / "images"
        self.trimaps_dir = self.data_dir / "trimaps"
        self.gt_dir = self.data_dir / "gt_alpha"

        # Get list of images
        self.image_names = sorted([f.stem for f in self.images_dir.glob("*.png")])

        print(f"Initialized evaluator: {len(self.image_names)} images found")

    def compute_sad(self, alpha_pred: np.ndarray, alpha_gt: np.ndarray,
                    trimap: np.ndarray = None) -> float:
        """Compute Sum of Absolute Differences."""
        alpha_pred = self._normalize_alpha(alpha_pred)
        alpha_gt = self._normalize_alpha(alpha_gt)

        if trimap is not None:
            mask = self._get_unknown_mask(trimap)
            diff = np.abs(alpha_pred[mask] - alpha_gt[mask])
        else:
            diff = np.abs(alpha_pred - alpha_gt)

        return float(np.sum(diff))

    def compute_mse(self, alpha_pred: np.ndarray, alpha_gt: np.ndarray,
                    trimap: np.ndarray = None) -> float:
        """Compute Mean Squared Error."""
        alpha_pred = self._normalize_alpha(alpha_pred)
        alpha_gt = self._normalize_alpha(alpha_gt)

        if trimap is not None:
            mask = self._get_unknown_mask(trimap)
            diff = alpha_pred[mask] - alpha_gt[mask]
        else:
            diff = alpha_pred - alpha_gt

        return float(np.mean(diff ** 2))

    def compute_mad(self, alpha_pred: np.ndarray, alpha_gt: np.ndarray,
                    trimap: np.ndarray = None) -> float:
        """Compute Mean Absolute Difference."""
        alpha_pred = self._normalize_alpha(alpha_pred)
        alpha_gt = self._normalize_alpha(alpha_gt)

        if trimap is not None:
            mask = self._get_unknown_mask(trimap)
            diff = np.abs(alpha_pred[mask] - alpha_gt[mask])
        else:
            diff = np.abs(alpha_pred - alpha_gt)

        return float(np.mean(diff))

    def _normalize_alpha(self, alpha: np.ndarray) -> np.ndarray:
        """Normalize alpha to [0, 1] range."""
        if np.max(alpha) > 1.0:
            return alpha / 255.0
        return alpha

    def _get_unknown_mask(self, trimap: np.ndarray) -> np.ndarray:
        """Get mask of unknown region in trimap."""
        return (trimap != 0) & (trimap != 255)

    def evaluate_image(self, image_name: str, lambda_param: float = 100.0,
                      window_radius: int = 1, epsilon: float = 1e-7,
                      save_visualizations: bool = True) -> Dict:
        """Evaluate matting on a single image."""
        print(f"\nEvaluating: {image_name}")

        # Load data
        image_path = self.images_dir / f"{image_name}.png"
        trimap_path = self.trimaps_dir / f"{image_name}.png"
        gt_path = self.gt_dir / f"{image_name}.png"

        image = cv2.imread(str(image_path)) / 255.0
        trimap = cv2.imread(str(trimap_path), cv2.IMREAD_GRAYSCALE)
        gt_alpha = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) / 255.0

        if image is None or trimap is None or gt_alpha is None:
            raise ValueError(f"Failed to load data for {image_name}")

        # Run matting algorithm
        start_time = time.time()
        try:
            alpha_pred = solve_closed_form_matting(
                image, trimap, lambda_param=lambda_param,
                window_radius=window_radius, epsilon=epsilon
            )
            elapsed_time = time.time() - start_time
            success = True
            error_msg = None
        except Exception as e:
            elapsed_time = time.time() - start_time
            success = False
            error_msg = str(e)
            alpha_pred = np.zeros_like(gt_alpha)
            print(f"Error: {error_msg}")

        # Save predicted alpha
        alpha_save_path = self.alphas_dir / f"{image_name}_alpha.png"
        cv2.imwrite(str(alpha_save_path), (alpha_pred * 255).astype(np.uint8))

        # Compute metrics
        metrics = {}
        if success:
            metrics['sad_unknown'] = self.compute_sad(alpha_pred, gt_alpha, trimap)
            metrics['mse_unknown'] = self.compute_mse(alpha_pred, gt_alpha, trimap)
            metrics['mad_unknown'] = self.compute_mad(alpha_pred, gt_alpha, trimap)
            metrics['sad_full'] = self.compute_sad(alpha_pred, gt_alpha, None)

            unknown_mask = self._get_unknown_mask(trimap)
            metrics['unknown_percentage'] = float(np.sum(unknown_mask) / unknown_mask.size * 100)

            print(f"Metrics: SAD={metrics['sad_unknown']:.2f}, "
                  f"MSE={metrics['mse_unknown']:.6f}, MAD={metrics['mad_unknown']:.6f}")

        # Create visualizations
        if save_visualizations and success:
            self._create_visualizations(image_name, image, trimap,
                                       alpha_pred, gt_alpha, metrics)

        # Compile results
        results = {
            'image_name': image_name,
            'success': success,
            'error': error_msg,
            'time_seconds': elapsed_time,
            'parameters': {
                'lambda': lambda_param,
                'window_radius': window_radius,
                'epsilon': epsilon
            },
            'metrics': metrics,
        }

        return results

    def _create_visualizations(self, image_name: str, image: np.ndarray,
                               trimap: np.ndarray, alpha_pred: np.ndarray,
                               gt_alpha: np.ndarray, metrics: Dict):
        """Create visualizations for a single image."""

        image_vis = (image * 255).astype(np.uint8)

        # 1. Main comparison grid
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{image_name} - Matting Evaluation', fontsize=14)

        # Row 1: Input, Trimap, Ground Truth
        axes[0, 0].imshow(cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Input Image')
        axes[0, 0].axis('off')

        # Trimap visualization
        trimap_vis = np.zeros((*trimap.shape, 3), dtype=np.uint8)
        trimap_vis[trimap == 255] = [255, 0, 0]  # Red = Foreground
        trimap_vis[trimap == 0] = [0, 0, 255]    # Blue = Background
        trimap_vis[(trimap != 0) & (trimap != 255)] = [128, 128, 128]  # Gray = Unknown
        axes[0, 1].imshow(trimap_vis)
        axes[0, 1].set_title('Trimap')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(gt_alpha, cmap='gray', vmin=0, vmax=1)
        axes[0, 2].set_title('Ground Truth Alpha')
        axes[0, 2].axis('off')

        # Row 2: Predicted Alpha, Error Map, Histogram
        axes[1, 0].imshow(alpha_pred, cmap='gray', vmin=0, vmax=1)
        axes[1, 0].set_title('Predicted Alpha')
        axes[1, 0].axis('off')

        # Error map on unknown region
        error = np.abs(alpha_pred - gt_alpha)
        unknown_mask = self._get_unknown_mask(trimap)
        error_vis = np.zeros_like(error)
        error_vis[unknown_mask] = error[unknown_mask]

        im = axes[1, 1].imshow(error_vis, cmap='jet', vmin=0, vmax=0.3)
        axes[1, 1].set_title(f'Error Map\nSAD={metrics["sad_unknown"]:.1f}')
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

        # Error histogram
        error_unknown = error[unknown_mask]
        axes[1, 2].hist(error_unknown, bins=50, color='steelblue', alpha=0.7)
        axes[1, 2].set_xlabel('Absolute Error')
        axes[1, 2].set_ylabel('Pixel Count')
        axes[1, 2].set_title(f'Error Distribution\nMAD={metrics["mad_unknown"]:.4f}')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.viz_dir / f"{image_name}_comparison.png", dpi=100)
        plt.close()

        # 2. Composite visualizations
        self._create_composite_visualization(image_name, image_vis,
                                            alpha_pred, gt_alpha)

        # 3. Error analysis
        self._create_error_analysis(image_name, alpha_pred, gt_alpha, trimap)

    def _create_composite_visualization(self, image_name: str, image: np.ndarray,
                                       alpha_pred: np.ndarray, gt_alpha: np.ndarray):
        """Create composite visualizations on different backgrounds."""

        backgrounds = [
            ('White', np.array([255, 255, 255])),
            ('Black', np.array([0, 0, 0])),
            ('Green', np.array([0, 255, 0])),
            ('Checkerboard', None)
        ]

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'{image_name} - Composites', fontsize=14)

        for col, (bg_name, bg_color) in enumerate(backgrounds):
            # Create background
            if bg_name == 'Checkerboard':
                bg = self._create_checkerboard(image.shape[:2])
            else:
                bg = np.ones_like(image) * bg_color

            # Composite with predicted alpha
            comp_pred = composite_on_background(image, alpha_pred, bg)
            axes[0, col].imshow(cv2.cvtColor(comp_pred, cv2.COLOR_BGR2RGB))
            axes[0, col].set_title(f'Predicted on {bg_name}')
            axes[0, col].axis('off')

            # Composite with ground truth alpha
            comp_gt = composite_on_background(image, gt_alpha, bg)
            axes[1, col].imshow(cv2.cvtColor(comp_gt, cv2.COLOR_BGR2RGB))
            axes[1, col].set_title(f'Ground Truth on {bg_name}')
            axes[1, col].axis('off')

        plt.tight_layout()
        plt.savefig(self.viz_dir / f"{image_name}_composites.png", dpi=100)
        plt.close()

    def _create_checkerboard(self, shape: Tuple[int, int], square_size: int = 20) -> np.ndarray:
        """Create a checkerboard pattern background."""
        h, w = shape
        checkerboard = np.zeros((h, w, 3), dtype=np.uint8)

        for i in range(0, h, square_size):
            for j in range(0, w, square_size):
                if ((i // square_size) + (j // square_size)) % 2 == 0:
                    checkerboard[i:i+square_size, j:j+square_size] = [200, 200, 200]
                else:
                    checkerboard[i:i+square_size, j:j+square_size] = [100, 100, 100]

        return checkerboard

    def _create_error_analysis(self, image_name: str, alpha_pred: np.ndarray,
                               gt_alpha: np.ndarray, trimap: np.ndarray):
        """Create error analysis visualization."""

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{image_name} - Error Analysis', fontsize=14)

        error = np.abs(alpha_pred - gt_alpha)
        unknown_mask = self._get_unknown_mask(trimap)

        # 1. Error by alpha value ranges
        alpha_bins = np.linspace(0, 1, 11)
        bin_centers = (alpha_bins[:-1] + alpha_bins[1:]) / 2
        mean_errors = []

        for i in range(len(alpha_bins) - 1):
            mask = unknown_mask & (gt_alpha >= alpha_bins[i]) & (gt_alpha < alpha_bins[i+1])
            if np.sum(mask) > 0:
                mean_errors.append(np.mean(error[mask]))
            else:
                mean_errors.append(0)

        axes[0].bar(bin_centers, mean_errors, width=0.08, color='coral', alpha=0.7)
        axes[0].set_xlabel('Ground Truth Alpha Value')
        axes[0].set_ylabel('Mean Absolute Error')
        axes[0].set_title('Error vs Alpha Value')
        axes[0].grid(True, alpha=0.3, axis='y')

        # 2. Spatial error distribution
        error_vis = np.zeros_like(error)
        error_vis[unknown_mask] = error[unknown_mask]
        im = axes[1].imshow(error_vis, cmap='hot',
                           vmin=0, vmax=np.percentile(error[unknown_mask], 95))
        axes[1].set_title('Spatial Error Distribution')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        plt.tight_layout()
        plt.savefig(self.viz_dir / f"{image_name}_error_analysis.png", dpi=100)
        plt.close()

    def run_evaluation(self, lambda_param: float = 100.0,
                      window_radius: int = 1, epsilon: float = 1e-7,
                      save_visualizations: bool = True) -> List[Dict]:
        """Evaluate matting on entire dataset."""
        print(f"\nEvaluating dataset: {len(self.image_names)} images")
        print(f"Parameters: lambda={lambda_param}, window={window_radius}, epsilon={epsilon}")

        results = []
        for image_name in self.image_names:
            result = self.evaluate_image(
                image_name, lambda_param=lambda_param,
                window_radius=window_radius, epsilon=epsilon,
                save_visualizations=save_visualizations
            )
            results.append(result)

        # Save summary
        self.save_summary(results)

        return results

    def save_summary(self, results: List[Dict]):
        """Save summary CSV with results."""
        import csv

        csv_path = self.output_dir / "results_summary.csv"

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'Image', 'Success', 'Time (s)',
                'SAD (unknown)', 'MSE (unknown)', 'MAD (unknown)',
                'SAD (full)', 'Unknown %'
            ])

            # Data rows
            for r in results:
                if r['success']:
                    m = r['metrics']
                    writer.writerow([
                        r['image_name'],
                        'Yes',
                        f"{r['time_seconds']:.2f}",
                        f"{m['sad_unknown']:.2f}",
                        f"{m['mse_unknown']:.6f}",
                        f"{m['mad_unknown']:.6f}",
                        f"{m['sad_full']:.2f}",
                        f"{m['unknown_percentage']:.1f}"
                    ])
                else:
                    writer.writerow([
                        r['image_name'], 'No', f"{r['time_seconds']:.2f}",
                        'N/A', 'N/A', 'N/A', 'N/A', 'N/A'
                    ])

        print(f"\nSaved CSV summary to: {csv_path}")

        # Print summary statistics
        successful = [r for r in results if r['success']]
        if successful:
            sads = [r['metrics']['sad_unknown'] for r in successful]
            mses = [r['metrics']['mse_unknown'] for r in successful]
            mads = [r['metrics']['mad_unknown'] for r in successful]
            times = [r['time_seconds'] for r in successful]

            print(f"\nEvaluation Summary:")
            print(f"Total: {len(results)}, Successful: {len(successful)}")
            print(f"SAD - mean: {np.mean(sads):.2f}, median: {np.median(sads):.2f}")
            print(f"MSE - mean: {np.mean(mses):.6f}, median: {np.median(mses):.6f}")
            print(f"MAD - mean: {np.mean(mads):.6f}, median: {np.median(mads):.6f}")
            print(f"Avg time: {np.mean(times):.2f} seconds")


def main():
    """Main evaluation script."""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate closed-form matting on dataset')
    parser.add_argument('--data_dir', type=str, default='../data',
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='../results/benchmark',
                       help='Path to save results')
    parser.add_argument('--lambda_param', type=float, default=100.0,
                       help='Lambda parameter for matting')
    parser.add_argument('--window_radius', type=int, default=1,
                       help='Window radius for Laplacian')
    parser.add_argument('--epsilon', type=float, default=1e-7,
                       help='Regularization parameter')
    parser.add_argument('--no_viz', action='store_true',
                       help='Skip visualization generation')

    args = parser.parse_args()

    # Create evaluator
    evaluator = MattingEvaluator(args.data_dir, args.output_dir)

    # Run evaluation
    results = evaluator.run_evaluation(
        lambda_param=args.lambda_param,
        window_radius=args.window_radius,
        epsilon=args.epsilon,
        save_visualizations=not args.no_viz
    )

    print(f"\nEvaluation complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
