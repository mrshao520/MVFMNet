<div align="center">

# [IEEE JMASS] MVFMNet: A Lightweight Network for Remote Sensing Image Super-Resolution

Wei Xue, Tiancheng Shao, Mingyang Du, Jing Zhou, Xiao Zheng and Ping Zhong

</div>

---
<p align="center">
  <img width="800" src="./assets/MVFMNet.png">
</p>

***Overview framework of the proposed method MVFMNet**. It consists of a shallow feature extraction module, a sequence of FMM, and a lightweight image reconstruction module, where the FMM contains a MVFMB and a SGFN.*

---
## Requirements
> - Python 3.8, PyTorch >= 1.8
> - BasicSR 1.4.2
> - Platforms: Ubuntu 22.04, cuda-11

## How To Test
- Refer to `./options/test` for the configuration file of the model to be tested, and prepare the testing data and pretrained model.
- The pretrained models are available in `./pretrained_models/`
- Then run the follwing codes (taking `MVFMNet_DF2K_x4SR.pth` as an example):

```
python basicsr/test.py -opt options/test/MVFMNet/MVFMNet_DF2K_x4SR.yml
```
The testing results will be saved in the `./results` folder.

## How To Train
- Refer to `./options/train` for the configuration file of the model to train.
- Preparation of training data can refer to this page. All datasets can be downloaded at the official website.
- The training command is like
```
python basicsr/train.py -opt options/train/MVFMNet/MVFMNet_DF2K_x4SR.yml
```
For more training commands and details, please check the docs in [BasicSR](https://github.com/XPixelGroup/BasicSR)  

## Citation
If you find this repository helpful, you may cite:
```
@ARTICLE{MVFMNet,
  author={Xue, Wei and Shao, Tiancheng and Du, Mingyang and Zhou, Jing and Zheng, Xiao},
  journal={IEEE Journal on Miniaturization for Air and Space Systems}, 
  title={MVFMNet: A Lightweight Network for Remote Sensing Image Super-Resolution}, 
  year={2026},
  volume={7},
  number={1},
  pages={36-46}
}
```