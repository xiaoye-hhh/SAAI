# SAAI
The implementation of ICCV 2023 [Visible-Infrared Person Re-Identification via Semantic Alignment and Affinity Inference](https://openaccess.thecvf.com/content/ICCV2023/papers/Fang_Visible-Infrared_Person_Re-Identification_via_Semantic_Alignment_and_Affinity_Inference_ICCV_2023_paper.pdf).

![框架](images/framework.png)

## :sparkles: News
- 2023-8-26: Release codes and our pretrained models at [Baidudisk](https://pan.baidu.com/s/1esd14VJturDXEK0vF2UUnw?pwd=7p3w) (7p3w).


## Getting Started

### Testing

1. Download the dataset and pretrained models (checkpoints) from [Baidudisk](https://pan.baidu.com/s/1esd14VJturDXEK0vF2UUnw?pwd=7p3w) (7p3w), unzip them.
2. Download the training data SYSU, unzip and put it in correct position.
3. Download the [rand_perm_cam.mat](https://github.com/wuancong/SYSU-MM01/tree/master/evaluation/data_split), and put it in correct position.
4. Change the dataset path in the file `configs/default/dataset.py`
5. Run the following command to retrain the model. You need about 22G GPU for the training.

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
```
numpy==1.24.2
torch==2.0.0
torchvision==0.15.1
PyYAML==6.0
pytorch-ignite==0.1.2
```


## Citation
If you find our work useful for your research, please consider citing the following papers :)

```bibtex
@InProceedings{Fang_2023_ICCV,
    author    = {Fang, Xingye and Yang, Yang and Fu, Ying},
    title     = {Visible-Infrared Person Re-Identification via Semantic Alignment and Affinity Inference},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {11270-11279}
}
``` 

## Contact
If you find any problem, please feel free to contact me (fangxingye@bit.edu.cn). A brief self-introduction is required, if you would like to get an in-depth help from me.