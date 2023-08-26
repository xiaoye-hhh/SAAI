# SAAI
The implementation of ICCV 2023 "[Visible-Infrared Person Re-Identification via Semantic Alignment and Affinity Inference]()".

![框架](images/framework.png)

## :sparkles: News
- 2023-8-26: Release codes and our pretrained models at [Baidudisk](https://pan.baidu.com/s/15dLeBPIzGpi7GG0TMUzkkg?pwd=iccv) (r8vb).


## Getting Started

### Testing

1. Download the dataset and pretrained models (checkpoints) from [Baidudisk](https://pan.baidu.com/s15dLeBPIzGpi7GG0TMUzkkg?pwd=iccv) (r8vb), unzip them.
2. Download the training data SYSU, unzip and put it in correct position.
3. Change the dataset path in the file `configs/default/dataset.py`
4. Run the following command to retrain the model. You need about 22G GPU for the training.

```shell
chmod 755 test.sh
./test
```

### Training

1. Download the training data SYSU, unzip and put it in correct position.
2. Change the dataset path in the file `configs/default/dataset.py`
3. Run the following command to retrain the model. You need about 22G GPU for the training.

```shell
chmod 755 train.sh
./train
```

## Requirement


## Citation
If you find our work useful for your research, please consider citing the following papers :)

<!-- ```bibtex
@inproceedings{fang2023saai,
  title={Visible-Infrared Person Re-Identification via Semantic Alignment and Affinity Inferenc},
  author={Xingye Fang, Yang Yang, and Ying Fu},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={},
  year={2023}
}
``` -->

## Contact
If you find any problem, please feel free to contact me (fangxingye@bit.edu.cn). A brief self-introduction is required, if you would like to get an in-depth help from me.