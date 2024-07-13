# Made to Order: Discovering monotonic temporal changes via self-supervised video ordering

Charig Yang, Weidi Xie, Andrew Zisserman

ECCV, 2024

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

Other datasets, see https://drive.google.com/file/d/1y0_2H_oCT4ixIGhmK64AlJzIYxHxId4W/view?usp=sharing

To run this on your own dataset, simply create a dataloader of the same nature.

## Citation
If you find this repository helpful, please consider citing our work:
```
@InProceedings{yang2024made,
      title={Made to Order: Discovering monotonic temporal changes via self-supervised video ordering}, 
      author={Charig Yang and Weidi Xie and Andrew Zisserman},
      booktitle={ECCV},
      year={2024},
}
```



 

 

