# Benchmark Results

Tested on 10 images from alphamatting.com dataset

## Results Summary

| Image | MAD | SAD | Time (s) |
|-------|-----|-----|----------|
| GT01  | 0.0293 | 1130.70 | 63.75 |
| GT02  | 0.0750 | 4309.37 | 65.12 |
| GT03  | 0.0468 | 5659.83 | 99.65 |
| GT04  | 0.0677 | 12097.73 | 73.30 |
| GT05  | 0.0430 | 1263.74 | 71.27 |
| GT07  | 0.0406 | 2150.79 | 89.47 |
| GT10  | 0.0521 | 2684.70 | 79.64 |
| GT15  | 0.0563 | 2450.89 | 60.79 |
| GT20  | 0.0276 | 1356.44 | 76.62 |
| GT26  | 0.1179 | 18061.40 | 84.77 |
| **Average** | **0.0556** | **5116.56** | **76.44** |

MAD = Mean Absolute Difference (lower is better)
SAD = Sum of Absolute Differences

## Parameters Used

- Lambda: 100
- Window radius: 1 (3x3 window)
- Epsilon: 1e-7
