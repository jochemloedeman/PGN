## Data

The data folder should look as follows
```
data/
|–– cifar100/
|–– cifar10/
|–– clevr/
|–– dtd
|–– eurosat/
|–– flowers102/
|–– food101/
|–– oxford_pets
|–– resisc45/
|–– sun397
|–– svhn/
|–– ucf101/
|–– DATA.md
```

For the DTD, EuroSAT, Flowers102, Oxford Pets, SUN397 and UCF101 folders, follow the instructions by [Zhou et al.](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md)
For RESISC45, download images from [here](www.escience.cn/people/JunweiHan/NWPU-RESISC45.html) and organize like this:

```
resisc45/
|–– resisc45_json.json
|–– images/
|   |–– airplane/
|   |–– airport/
|   |–– ...
```
Finally for CLEVR/count, download the content from [here](https://cs.stanford.edu/people/jcjohns/clevr/) and run `create_clevr_annotations.py`. The directory should look like
```
clevr/
|–– clevr_count/
|–– CLEVR_v1.0
```
