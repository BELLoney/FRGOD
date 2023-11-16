# FRGOD
Zhong Yuan, **Hongmei Chen***, Tianrui Li, Binbin Sang, and Shu Wang, 
[Outlier Detection Based on Fuzzy Rough Granules in Mixed Attribute Data](FRGOD_code/2022-FRGOD.pdf), 
IEEE Transactions on Cybernetics, Volume: 52, Issue: 8, August 2022, 
DOI: [10.1109/TCYB.2021.3058780](https://doi.org/10.1109/TCYB.2021.3058780). (Code)

## Abstract
Outlier detection is one of the most important research directions in data mining. 
However, most of the current research focuses on outlier detection for categorical or numerical attribute data. 
There are few studies on the outlier detection of mixed attribute data. 
In this article, we introduce fuzzy rough sets (FRSs) to deal with the problem of outlier detection in mixed attribute data. 
Since the outlier detection model of the classical rough set is only applicable to the categorical attribute data, we use FRS to generalize the outlier detection model and construct a generalized outlier detection model based on fuzzy rough granules. 
First, the granule outlier degree (GOD) is defined to characterize the outlier degree of fuzzy rough granules by employing the fuzzy approximation accuracy. 
Then, the outlier factor based on fuzzy rough granules is constructed by integrating the GOD and the corresponding weights to characterize the outlier degree of objects. 
Furthermore, the corresponding fuzzy rough granules-based outlier detection (FRGOD) algorithm is designed. The effectiveness of the FRGOD algorithm is evaluated through experiments on 16 real-world datasets. 
The experimental results show that the algorithm is more flexible for detecting outliers and is suitable for numerical, categorical, and mixed attribute data.

## Usage
You can run Demo_FRGOD.m or FRGOD.py:
```
clc;
clear all

load Example.mat
Dataori=Example;
format shortG

trandata=Dataori;
trandata(:,2:3)=normalize(trandata(:,2:3),'range');

lam=1;
out_factors=FRGOD(trandata,lam)

```
You can get outputs as follows:
```
out_factors =
      0.31923
      0.18569
      0.19614
      0.18808
      0.16553
      0.18453
```

## Citation
If you find FRGOD useful in your research, please consider citing:
```
@article{yuan2022outlier,
  title={Outlier Detection Based on Fuzzy Rough Granules in Mixed Attribute Data},
  author={Yuan, Zhong and Chen, Hong Mei and Li, Tian Rui and Sang, Bin Bin and Wang, Shu},
  journal={IEEE Transactions on Cybernetics},
  volume={52},
  pages={8399--8412},
  year={2022},
  doi={10.1109/TCYB.2021.3058780},
  publisher={IEEE}
}
```