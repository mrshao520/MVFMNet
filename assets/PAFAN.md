<div align="center">

# [IEEE GRSL] Partial Attention Feature Aggregation Network for Lightweight Remote Sensing Image Super-Resolution

Wei Xue, Tiancheng Shao, Mingyang Du, Xiao Zheng and Ping Zhong

</div>

---

<p align="center">
  <img width="800" src="./PAFAN.png">
</p>

***Overall architecture of the proposed partial attention feature aggregation network (PAFAN)**. It contains four main stages, i.e., shallow feature extraction, a sequence of attention progressive feature distillation blocks, multi-layer feature fusion, and image reconstruction.*

---

## Requirements

> - Python 3.8, PyTorch >= 1.8
> - BasicSR 1.4.2
> - Platforms: Ubuntu 22.04, cuda-11

## How To Test

- Refer to `./options/test` for the configuration file of the model to be tested, and prepare the testing data and pretrained model.
- The pretrained models are available in `./pretrained_models/`
- Then run the follwing codes (taking `PAFAN_UC_x4SR.pth` as an example):

```
python basicsr/test.py -opt options/test/PAFAN/PAFAN_UC_x4SR.yml
```

The testing results will be saved in the `./results` folder.

## How To Train

- Refer to `./options/train` for the configuration file of the model to train.
- Preparation of training data can refer to this page. All datasets can be downloaded at the official website.
- The training command is like

```
python basicsr/train.py -opt options/train/PAFAN/PAFAN_UC_x4SR.yml
```

For more training commands and details, please check the docs in [BasicSR](https://github.com/XPixelGroup/BasicSR)  

## Citation

If you find this repository helpful, you may cite:

```
@ARTICLE{PAFAN,
  author={Xue, Wei and Shao, Tiancheng and Du, Mingyang and Zheng, Xiao and Zhong, Ping},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Partial Attention Feature Aggregation Network for Lightweight Remote Sensing Image Super-Resolution}, 
  year={2025},
  volume={22},
  pages={1-5}
}
```

