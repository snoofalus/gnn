# Graph convolutional Semi-supervised node classification for multimodal image segmentation
The code of this repository is an unofficial project exploring the use of chebyshev graph convolutions  [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375) and non-chebyshev convolutions
[Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) for node classification in semantic segmentation of multimodal images.

## Prerequisites
- pytorch (1.8+ as 1.7 and below does not have same sparse functionality)
- numpy
- scipy
- matplotlib

## Usage
To use with a different dataset the model expects:  
A folder of multimodal numpy images @ ../data-local/images/*dataset*/data  
Numpy masks with the same name as their corresponding image @ ../data-local/images/*my_dataset*/mask  
A new dataset loader file using the custom pytorch loaders @ /dataset/*my_dataset*.py  

The model can then be run with default arguments as
```python
python train.py
```

## References
```
@article{DBLP:journals/corr/DefferrardBV16,
  author    = {Micha{\"{e}}l Defferrard and
               Xavier Bresson and
               Pierre Vandergheynst},
  title     = {Convolutional Neural Networks on Graphs with Fast Localized Spectral
               Filtering},
  journal   = {CoRR},
  volume    = {abs/1606.09375},
  year      = {2016},
  url       = {http://arxiv.org/abs/1606.09375},
  archivePrefix = {arXiv},
  eprint    = {1606.09375},
  timestamp = {Mon, 13 Aug 2018 16:48:03 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/DefferrardBV16.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

```
@article{DBLP:journals/corr/KipfW16,
  author    = {Thomas N. Kipf and
               Max Welling},
  title     = {Semi-Supervised Classification with Graph Convolutional Networks},
  journal   = {CoRR},
  volume    = {abs/1609.02907},
  year      = {2016},
  url       = {http://arxiv.org/abs/1609.02907},
  archivePrefix = {arXiv},
  eprint    = {1609.02907},
  timestamp = {Mon, 13 Aug 2018 16:48:31 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/KipfW16.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```