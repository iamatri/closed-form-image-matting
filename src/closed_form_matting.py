"""
Closed-Form Image Matting
Based on Levin et al. 2006
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import cv2
import warnings
from typing import Tuple, Optional

from matting_laplacian import compute_matting_laplacian_fast


def solve_closed_form_matting(
    image: np.ndarray,
    constraints: np.ndarray,
    lambda_param: float = 100.0,
    window_radius: int = 1,
    epsilon: float = 1e-7,
    use_confidence: bool = False,
    confidence_map: Optional[np.ndarray] = None
) -> np.ndarray:
    """Solve for alpha matte using closed-form matting"""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must be (H, W, 3) array")

    if constraints.ndim == 3:
        constraints = constraints[:, :, 0]

    if image.shape[:2] != constraints.shape:
        raise ValueError(f"Image and constraints must have same size. "
                        f"Got image={image.shape[:2]}, constraints={constraints.shape}")

    if np.max(image) > 1.0:
        image = image / 255.0

    h, w = image.shape[:2]
    n_pixels = h * w

    print(f"solving matting for {w}x{h} image, lambda={lambda_param}, window={window_radius}")
    print(f"computing laplacian...")
    L = compute_matting_laplacian_fast(image, window_radius=window_radius, epsilon=epsilon)
    print(f"laplacian: {L.shape}, {L.nnz} nonzeros")

    constraints_flat = constraints.reshape(n_pixels)

    fg_mask = (constraints_flat == 255)
    bg_mask = (constraints_flat == 0)
    known_mask = fg_mask | bg_mask
    unknown_mask = ~known_mask

    n_known = np.sum(known_mask)
    n_unknown = np.sum(unknown_mask)

    print(f"constraints: fg={np.sum(fg_mask)}, bg={np.sum(bg_mask)}, unknown={n_unknown}")

    b = np.zeros(n_pixels)
    b[fg_mask] = 1.0
    b[bg_mask] = 0.0

    if use_confidence and confidence_map is not None:
        confidence_flat = confidence_map.reshape(n_pixels)
        d_values = np.zeros(n_pixels)
        d_values[known_mask] = confidence_flat[known_mask]
    else:
        d_values = known_mask.astype(float)

    D = sparse.diags(d_values, 0, shape=(n_pixels, n_pixels), format='csr')

    print(f"building system and solving...")
    A = L + lambda_param * D
    rhs = lambda_param * b

    try:
        alpha_flat = spsolve(A, rhs)
    except Exception as e:
        print(f"warning: direct solve failed, adding regularization...")
        A_reg = A + 1e-6 * sparse.eye(n_pixels)
        alpha_flat = spsolve(A_reg, rhs)

    alpha = alpha_flat.reshape(h, w)
    alpha = np.clip(alpha, 0, 1)

    print(f"done! alpha range: [{alpha.min():.3f}, {alpha.max():.3f}]")

    return alpha


def extract_foreground(
    image: np.ndarray,
    alpha: np.ndarray,
    return_rgb: bool = True
) -> np.ndarray:
    """Extract foreground using alpha matte"""
    if image.shape[:2] != alpha.shape:
        raise ValueError("Image and alpha must have same dimensions")

    if np.max(image) <= 1.0:
        image = (image * 255).astype(np.uint8)

    foreground = image.copy().astype(float)
    alpha_3ch = np.stack([alpha, alpha, alpha], axis=2)

    if return_rgb:
        foreground = foreground * alpha_3ch
        return foreground.astype(np.uint8)
    else:
        rgba = np.concatenate([
            foreground.astype(np.uint8),
            (alpha * 255).astype(np.uint8)[:, :, np.newaxis]
        ], axis=2)
        return rgba


def composite_on_background(
    foreground: np.ndarray,
    alpha: np.ndarray,
    background: np.ndarray
) -> np.ndarray:
    """Composite foreground onto background using alpha matte"""
    if foreground.shape[:2] != alpha.shape:
        raise ValueError("Foreground and alpha must have same dimensions")

    h, w = foreground.shape[:2]

    if background.shape == (3,) or background.shape == (1, 1, 3):
        background = np.ones((h, w, 3)) * background.reshape(1, 1, 3)
    elif background.shape[:2] != (h, w):
        background = cv2.resize(background, (w, h))

    fg = foreground.copy().astype(float)
    bg = background.copy().astype(float)
    if np.max(fg) > 1.0:
        fg = fg / 255.0
    if np.max(bg) > 1.0:
        bg = bg / 255.0

    alpha_3ch = np.stack([alpha, alpha, alpha], axis=2)
    composite = alpha_3ch * fg + (1 - alpha_3ch) * bg
    composite = np.clip(composite * 255, 0, 255).astype(np.uint8)

    return composite


def refine_trimap(
    trimap: np.ndarray,
    erosion_size: int = 5,
    dilation_size: int = 5
) -> np.ndarray:
    """Refine trimap by eroding foreground and dilating background"""
    fg_mask = (trimap == 255)
    bg_mask = (trimap == 0)

    if erosion_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
        fg_mask = cv2.erode(fg_mask.astype(np.uint8), kernel).astype(bool)

    if dilation_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
        bg_mask = cv2.dilate(bg_mask.astype(np.uint8), kernel).astype(bool)

    refined = np.full(trimap.shape, 128, dtype=np.uint8)
    refined[fg_mask] = 255
    refined[bg_mask] = 0

    overlap = fg_mask & bg_mask
    if np.any(overlap):
        refined[overlap] = 128

    return refined


def trimap_from_alpha(
    alpha: np.ndarray,
    fg_threshold: float = 0.95,
    bg_threshold: float = 0.05
) -> np.ndarray:
    """Create trimap from alpha matte"""
    trimap = np.full(alpha.shape, 128, dtype=np.uint8)
    trimap[alpha >= fg_threshold] = 255
    trimap[alpha <= bg_threshold] = 0
    return trimap


def visualize_matting_results(
    image: np.ndarray,
    alpha: np.ndarray,
    constraints: Optional[np.ndarray] = None,
    ground_truth: Optional[np.ndarray] = None,
    background_color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """Create visualization showing matting results"""
    if np.max(image) <= 1.0:
        image = (image * 255).astype(np.uint8)

    h, w = image.shape[:2]
    images_to_show = []

    img_with_title = image.copy()
    cv2.putText(img_with_title, "Input Image", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    images_to_show.append(img_with_title)

    if constraints is not None:
        trimap_viz = np.zeros((h, w, 3), dtype=np.uint8)
        trimap_viz[constraints == 255] = [0, 0, 255]
        trimap_viz[constraints == 0] = [255, 0, 0]
        trimap_viz[(constraints != 0) & (constraints != 255)] = [128, 128, 128]

        cv2.putText(trimap_viz, "Constraints", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        images_to_show.append(trimap_viz)

    alpha_viz = (alpha * 255).astype(np.uint8)
    alpha_viz = cv2.cvtColor(alpha_viz, cv2.COLOR_GRAY2BGR)
    cv2.putText(alpha_viz, "Alpha Matte", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    images_to_show.append(alpha_viz)

    bg = np.array(background_color).reshape(1, 1, 3)
    composite = composite_on_background(image, alpha, bg)
    cv2.putText(composite, "Composite", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    images_to_show.append(composite)

    if ground_truth is not None:
        gt_viz = (ground_truth * 255 if np.max(ground_truth) <= 1.0 else ground_truth).astype(np.uint8)
        gt_viz = cv2.cvtColor(gt_viz, cv2.COLOR_GRAY2BGR)
        cv2.putText(gt_viz, "Ground Truth", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        images_to_show.append(gt_viz)

        alpha_norm = alpha if np.max(alpha) <= 1.0 else alpha / 255.0
        gt_norm = ground_truth if np.max(ground_truth) <= 1.0 else ground_truth / 255.0
        error = np.abs(alpha_norm - gt_norm)

        error_viz = cv2.applyColorMap((error * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.putText(error_viz, f"Error (SAD={np.sum(error):.1f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        images_to_show.append(error_viz)

    n_images = len(images_to_show)
    if n_images <= 3:
        grid = np.hstack(images_to_show)
    else:
        n_per_row = (n_images + 1) // 2
        row1 = np.hstack(images_to_show[:n_per_row])
        row2 = np.hstack(images_to_show[n_per_row:])

        if row2.shape[1] < row1.shape[1]:
            pad_width = row1.shape[1] - row2.shape[1]
            row2 = np.hpad(row2, ((0, 0), (0, pad_width), (0, 0)),
                          mode='constant', constant_values=0)

        grid = np.vstack([row1, row2])

    return grid


def compute_matting_metrics(
    alpha_pred: np.ndarray,
    alpha_gt: np.ndarray,
    trimap: Optional[np.ndarray] = None
) -> dict:
    """Compute evaluation metrics for alpha matte"""
    alpha_pred = alpha_pred / 255.0 if np.max(alpha_pred) > 1.0 else alpha_pred
    alpha_gt = alpha_gt / 255.0 if np.max(alpha_gt) > 1.0 else alpha_gt

    if alpha_pred.shape != alpha_gt.shape:
        raise ValueError("Predicted and ground truth alpha must have same shape")

    if trimap is not None:
        unknown_mask = (trimap != 0) & (trimap != 255)
        pred_eval = alpha_pred[unknown_mask]
        gt_eval = alpha_gt[unknown_mask]
        eval_region = "unknown"
    else:
        pred_eval = alpha_pred.flatten()
        gt_eval = alpha_gt.flatten()
        eval_region = "full"

    diff = pred_eval - gt_eval
    abs_diff = np.abs(diff)

    metrics = {
        'sad': np.sum(abs_diff),
        'mse': np.mean(diff ** 2),
        'mad': np.mean(abs_diff),
        'region': eval_region,
        'n_pixels': len(pred_eval)
    }

    return metrics


if __name__ == "__main__":
    print("testing closed-form matting implementation")
    print("creating synthetic test image...")

    h, w = 100, 100
    image = np.zeros((h, w, 3))
    for i in range(3):
        image[:, :, i] = np.linspace(0, 1, w)

    trimap = np.full((h, w), 128, dtype=np.uint8)
    center = (h // 2, w // 2)
    for y in range(h):
        for x in range(w):
            dist = np.sqrt((y - center[0])**2 + (x - center[1])**2)
            if dist < 20:
                trimap[y, x] = 255
            elif dist > 40:
                trimap[y, x] = 0

    print(f"image: {image.shape}, fg={np.sum(trimap == 255)}, bg={np.sum(trimap == 0)}, unknown={np.sum((trimap != 0) & (trimap != 255))}")

    print("solving for alpha matte...")
    alpha = solve_closed_form_matting(
        image,
        trimap,
        lambda_param=100.0,
        window_radius=1
    )

    print(f"alpha computed! range=[{alpha.min():.3f}, {alpha.max():.3f}], mean={alpha.mean():.3f}")

    print("testing foreground extraction...")
    fg_rgb = extract_foreground(image, alpha, return_rgb=True)
    fg_rgba = extract_foreground(image, alpha, return_rgb=False)
    print(f"fg shapes: rgb={fg_rgb.shape}, rgba={fg_rgba.shape}")

    print("testing compositing...")
    bg_green = np.array([0, 255, 0])
    composite = composite_on_background(image, alpha, bg_green)
    print(f"composite shape: {composite.shape}")

    print("test complete!")
