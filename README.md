# GCformer

GCformer combines a structured global convolutional branch for processing long input sequences with a local Transformer-based branch for capturing short, recent signals.
Experiments demonstrate that GCformer outperforms state-of-the-art methods, reducing MSE error in multivariate time series benchmarks by 4.38\% and model parameters by 61.92\%. In particular, the global convolutional branch can serve as a plug-in block to enhance the performance of other models, with an average improvement of 31.93\%, including various recently published Transformer-based models.

## Method

|![model_structure](https://github.com/zyj-111/GCformer/assets/52376036/6a0ccb38-7600-418d-a0ad-39740ba76773)|
|:--:| 
| *Figure 1. GCformer overall framework* |

|![global_kernel](https://github.com/zyj-111/GCformer/assets/52376036/5ea13caf-6fa9-4f40-a205-beee422e0d6d)|
|:--:| 
| *Figure 2. Different parameterization methods of global convolution kernel* |

## Main Results
|![boosting_result](https://github.com/zyj-111/GCformer/assets/52376036/d0e8b7ba-aab1-40f9-aafe-ccd4e263e545)|

|![full_benchmark](https://github.com/zyj-111/GCformer/assets/52376036/cf701b90-94e1-4415-b7cf-41484925591b)|

## Get Started

1. Install Python 3.6, PyTorch 1.11.0.
2. Download data. You can obtain all the six benchmarks from
[[FEDformer](https://github.com/MAZiqing/FEDformer)] or [[Autoformer](https://github.com/thuml/Autoformer)].
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts/GCformer`. For instance, you can reproduce the experiment result on illness dataset by:

```bash
bash ./scripts/GCformer/illness.sh
```

## Citation

## Contact

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/yuqinie98/PatchTST

https://github.com/MAZiqing/FEDformer

https://github.com/ctlllll/SGConv

https://github.com/thuml/Autoformer