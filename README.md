# NeurASP

Implementation of [NeurASP: Embracing Neural Networks into Answer Set Programming](https://www.ijcai.org/proceedings/2020/0243.pdf).

## Introduction

NeurASP extends Answer Set Programming (ASP) with neural predicates. Neural network outputs are treated as probabilistic facts, enabling a combined neuro-symbolic pipeline.

This repository demonstrates:
1. how symbolic rules improve downstream reasoning when combined with neural perception;
2. how neural models can be trained with semantic constraints, not only labeled supervision.

## Quick Start

Run these from the project root:
List of tasks that have been integrated with our code.

```bash
python3 -m examples.mnistAdd.mnist
python3 -m examples.add2x2.train
python3 -m examples.member3.train
python3 -m examples.member5.train
python3 -m examples.card_arithmetic.train --image_dir examples/card_arithmetic/data
python3 -m examples.card_arithmetic.train --image_dir examples/card_arithmetic/data --variant sum_3
python3 -m examples.card_arithmetic.train --image_dir examples/card_arithmetic/data --variant sum_4
```

## Debugging

If debugpy port 5678 is already occupied:

```bash
kill -9 $(lsof -t -i:5678)
```

Start training under debugpy:

```bash
python3 -m debugpy --listen 5678 --wait-for-client -m examples.mnistAdd.mnist
python3 -m debugpy --listen 5678 --wait-for-client -m examples.add2x2.train
python3 -m debugpy --listen 5678 --wait-for-client -m examples.member3.train
python3 -m debugpy --listen 5678 --wait-for-client -m examples.member5.train
python3 -m debugpy --listen 5678 --wait-for-client -m examples.card_arithmetic.train --image_dir examples/card_arithmetic/data
python3 -m debugpy --listen 5678 --wait-for-client -m examples.card_arithmetic.train --image_dir examples/card_arithmetic/data --variant sum_3
python3 -m debugpy --listen 5678 --wait-for-client -m examples.card_arithmetic.train --image_dir examples/card_arithmetic/data --variant sum_4
```

## Installation

1. Install Anaconda (or Miniconda):
   https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
2. Clone the repository:

```bash
git clone https://github.com/azreasoners/NeurASP
cd NeurASP
```

3. Create and activate an environment, then install core dependencies:

```bash
conda create --name neurasp python=3.9
conda activate neurasp
conda install -c potassco clingo=5.5 tqdm
```

4. Install PyTorch following the official selector:
   https://pytorch.org/get-started/locally/

Example (Linux + CUDA 10.2; PyTorch 1.12.0 tested in this project):

```bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

## Example Suite

This repository provides 3 inference examples and 10 learning examples. Each example folder contains task-specific notes.

### Inference Examples

- [Sudoku](https://github.com/azreasoners/NeurASP/tree/master/examples/sudoku)
- [Offset Sudoku](https://github.com/azreasoners/NeurASP/tree/master/examples/offset_sudoku)
- [Toy-car](https://github.com/azreasoners/NeurASP/tree/master/examples/toycar)

### Learning Examples

- [MNIST Addition](https://github.com/azreasoners/NeurASP/tree/master/examples/mnistAdd)
- [Shortest Path](https://github.com/azreasoners/NeurASP/tree/master/examples/shortest_path)
- [Sudoku Solving](https://github.com/azreasoners/NeurASP/tree/master/examples/solvingSudoku_70k)
- [Top-k](https://github.com/azreasoners/NeurASP/tree/master/examples/top_k)
- [Most Reliable Path](https://github.com/azreasoners/NeurASP/tree/master/examples/most_reliable_path)
- NeuroLog tasks: [add2x2](https://github.com/azreasoners/NeurASP/tree/master/examples/add2x2), [apply2x2](https://github.com/azreasoners/NeurASP/tree/master/examples/apply2x2), [member3](https://github.com/azreasoners/NeurASP/tree/master/examples/member3), [member5](https://github.com/azreasoners/NeurASP/tree/master/examples/member5)
- [Card arithmetic](https://github.com/azreasoners/NeurASP/tree/master/examples/card_arithmetic)

## Citation

Please cite the original NeurASP paper:

```bibtex
@inproceedings{ijcai2020p243,
  title     = {NeurASP: Embracing Neural Networks into Answer Set Programming},
  author    = {Yang, Zhun and Ishay, Adam and Lee, Joohyung},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Christian Bessiere},
  pages     = {1755--1762},
  year      = {2020},
  month     = {7},
  note      = {Main track},
  doi       = {10.24963/ijcai.2020/243},
  url       = {https://doi.org/10.24963/ijcai.2020/243},
}
```

For the follow-up paper on vectorization, caching, and card arithmetic:

```bibtex
@article{rader2026accelerating_neurasp,
  title   = {Accelerating NeurASP with Vectorization and Caching},
  DOI     = {10.1017/S1471068426100611},
  journal = {Theory and Practice of Logic Programming},
  author  = {Rader, Alexander Philipp and Russo, Alessandra},
  year    = {2026},
  pages   = {1-16}
}
```