# Made to Order: Discovering monotonic temporal changes via self-supervised video ordering

Charig Yang, Weidi Xie, Andrew Zisserman

Visual Geometry Group, Department of Engineering Science, University of Oxford

## Requirements
```pytorch```,
```opencv```,
```einops```,
```tensorboardX```


## How to use 

To get started, 
```
python main.py
```

This should train the model on MNIST under default settings. You may visualise the training and attribution maps on Tensorboard.

We have included a instructions on how to train on several datsets (RDS, MNIST and SVHN). Check `main.py`. The dataset should be downloaded automatically on the first run (or created on the fly, as in RDS). 

Other datasets will be uploaded in due course.

To run this on your own dataset, simply create a dataloader of the same nature.

## Citation
If you find this repository helpful, please consider citing our work:
```
@article{yang2024made,
      title={Made to Order: Discovering monotonic temporal changes via self-supervised video ordering}, 
      author={Charig Yang and Weidi Xie and Andrew Zisserman},
      year={2024},
      eprint={2404.16828},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



 

 

