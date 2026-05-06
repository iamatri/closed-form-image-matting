# Closed-Form Image Matting

CS 445 Final Project - Anurag Atri (aatri2), Hela Kasibhotla (helask2)

Implementation of "A Closed-Form Solution to Natural Image Matting" (Levin et al., CVPR 2006)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Try the interactive tool:
```bash
python src/interactive_matting.py data/images/GT01.png
```

Left click = foreground (red), right click = background (blue)
Press space to run the algorithm, s to save, q to quit

Run benchmark evaluation:
```bash
python src/eval/evaluate_benchmark.py --data_dir data --output_dir results/benchmark
```

## Results

Evaluated on 10 images from alphamatting.com benchmark
- Average MAD: 0.0556
- Successfully processed all images

See results/results_summary.md for details

## Code Structure

- src/closed_form_matting.py - main algorithm
- src/matting_laplacian.py - matting Laplacian computation
- src/interactive_matting.py - interactive scribble tool
- src/eval/ - evaluation scripts
- src/download_dataset.py - download benchmark dataset

## References

Paper: https://people.csail.mit.edu/alevin/papers/Matting-Levin-Lischinski-Weiss-CVPR06.pdf
Dataset: http://alphamatting.com
