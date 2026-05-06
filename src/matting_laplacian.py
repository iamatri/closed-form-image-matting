"""
Matting Laplacian Implementation
Based on Levin et al. 2006 paper
"""

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, diags
import warnings


def compute_matting_laplacian(image, window_radius=1, epsilon=1e-7):
    """Compute matting Laplacian matrix from image"""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must be (H, W, 3) array")

    if np.min(image) < 0 or np.max(image) > 1:
        warnings.warn("Image values should be in [0, 1] range. Clipping values.")
        image = np.clip(image, 0, 1)

    h, w, c = image.shape
    n_pixels = h * w
    window_size = (2 * window_radius + 1) ** 2

    img_reshaped = image.reshape(n_pixels, 3)

    row_inds = []
    col_inds = []
    values = []

    for y in range(h):
        for x in range(w):
            y_min = max(0, y - window_radius)
            y_max = min(h, y + window_radius + 1)
            x_min = max(0, x - window_radius)
            x_max = min(w, x + window_radius + 1)

            window_pixels = []
            window_indices = []

            for wy in range(y_min, y_max):
                for wx in range(x_min, x_max):
                    pixel_idx = wy * w + wx
                    window_pixels.append(img_reshaped[pixel_idx])
                    window_indices.append(pixel_idx)

            window_pixels = np.array(window_pixels)
            window_indices = np.array(window_indices)
            win_size = len(window_indices)

            if win_size == 0:
                continue

            mu_k = np.mean(window_pixels, axis=0)
            centered = window_pixels - mu_k
            sigma_k = (centered.T @ centered) / win_size
            regularization = (epsilon / win_size) * np.eye(3)
            sigma_reg = sigma_k + regularization

            try:
                inv_sigma = np.linalg.inv(sigma_reg)
            except np.linalg.LinAlgError:
                sigma_reg = sigma_k + (epsilon * 10) * np.eye(3)
                inv_sigma = np.linalg.inv(sigma_reg)

            for i_idx in range(win_size):
                i = window_indices[i_idx]
                I_i = window_pixels[i_idx] - mu_k

                for j_idx in range(win_size):
                    j = window_indices[j_idx]
                    I_j = window_pixels[j_idx] - mu_k

                    delta_ij = 1.0 if i == j else 0.0
                    affinity = I_i @ inv_sigma @ I_j
                    L_ij = (delta_ij / win_size) - (1.0 / (win_size ** 2)) * (1 + affinity)

                    row_inds.append(i)
                    col_inds.append(j)
                    values.append(L_ij)

    L = csr_matrix((values, (row_inds, col_inds)), shape=(n_pixels, n_pixels))

    return L


def compute_matting_laplacian_fast(image, window_radius=1, epsilon=1e-7):
    """Faster matting Laplacian using vectorized operations"""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must be (H, W, 3) array")

    if np.min(image) < 0 or np.max(image) > 1:
        warnings.warn("Image values should be in [0, 1] range. Clipping values.")
        image = np.clip(image, 0, 1)

    h, w, c = image.shape
    n_pixels = h * w
    window_size = (2 * window_radius + 1) ** 2

    max_entries = n_pixels * window_size ** 2
    row_inds = np.zeros(max_entries, dtype=np.int32)
    col_inds = np.zeros(max_entries, dtype=np.int32)
    values = np.zeros(max_entries, dtype=np.float64)

    entry_count = 0
    img_reshaped = image.reshape(n_pixels, 3)

    for y in range(h):
        for x in range(w):
            y_min = max(0, y - window_radius)
            y_max = min(h, y + window_radius + 1)
            x_min = max(0, x - window_radius)
            x_max = min(w, x + window_radius + 1)

            window_h = y_max - y_min
            window_w = x_max - x_min

            wy, wx = np.meshgrid(range(y_min, y_max), range(x_min, x_max), indexing='ij')
            window_indices = wy.ravel() * w + wx.ravel()
            win_size = len(window_indices)

            window_pixels = img_reshaped[window_indices]

            mu_k = np.mean(window_pixels, axis=0)
            centered = window_pixels - mu_k

            sigma_k = (centered.T @ centered) / win_size
            sigma_reg = sigma_k + (epsilon / win_size) * np.eye(3)

            try:
                inv_sigma = np.linalg.inv(sigma_reg)
            except np.linalg.LinAlgError:
                sigma_reg = sigma_k + (epsilon * 10) * np.eye(3)
                inv_sigma = np.linalg.inv(sigma_reg)

            affinity_matrix = centered @ inv_sigma @ centered.T
            identity = np.eye(win_size)
            L_window = (identity / win_size) - (1.0 / (win_size ** 2)) * (1 + affinity_matrix)

            for i_idx in range(win_size):
                for j_idx in range(win_size):
                    row_inds[entry_count] = window_indices[i_idx]
                    col_inds[entry_count] = window_indices[j_idx]
                    values[entry_count] = L_window[i_idx, j_idx]
                    entry_count += 1

    row_inds = row_inds[:entry_count]
    col_inds = col_inds[:entry_count]
    values = values[:entry_count]

    L = csr_matrix((values, (row_inds, col_inds)), shape=(n_pixels, n_pixels))

    return L


def verify_laplacian_properties(L, tolerance=1e-6):
    """Check if Laplacian matrix has expected properties"""
    results = {}

    diff = L - L.T
    max_asymmetry = np.max(np.abs(diff.data)) if diff.nnz > 0 else 0
    results['is_symmetric'] = max_asymmetry < tolerance
    results['max_asymmetry'] = max_asymmetry

    row_sums = np.array(L.sum(axis=1)).flatten()
    max_row_sum = np.max(np.abs(row_sums))
    results['row_sums_zero'] = max_row_sum < tolerance
    results['max_row_sum'] = max_row_sum

    results['nnz'] = L.nnz
    results['density'] = L.nnz / (L.shape[0] ** 2)

    return results


if __name__ == "__main__":
    print("testing matting laplacian implementation")

    np.random.seed(42)
    test_image = np.random.rand(10, 10, 3)

    print(f"test image: {test_image.shape}, range=[{test_image.min():.3f}, {test_image.max():.3f}]")

    print("computing laplacian (standard method)...")
    L_standard = compute_matting_laplacian(test_image, window_radius=1, epsilon=1e-7)

    print(f"laplacian: {L_standard.shape}, nnz={L_standard.nnz}, sparsity={L_standard.nnz / (L_standard.shape[0]**2) * 100:.2f}%")

    print("verifying laplacian properties...")
    props = verify_laplacian_properties(L_standard)

    for key, value in props.items():
        print(f"  {key}: {value}")

    print("computing laplacian (fast method)...")
    L_fast = compute_matting_laplacian_fast(test_image, window_radius=1, epsilon=1e-7)

    diff = L_standard - L_fast
    max_diff = np.max(np.abs(diff.data)) if diff.nnz > 0 else 0
    print(f"max difference between methods: {max_diff:.2e}")

    if max_diff < 1e-10:
        print("success: both methods produce identical results!")
    else:
        print("warning: methods produce different results")

    print("test complete!")
