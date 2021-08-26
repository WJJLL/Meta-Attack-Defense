# Meta-Attack-Defense
## Requirements：
- python 3.7.9
- CUDA==11.2
- Market1501,DukeMTMC-reID,MSMT-17,PersonX456,UnrealPerson,RandPerson
- torch==1.3.1
- torchvision==0.2.1
## Preparing Data
- Market1501,DukeMTMC-reID,MSMT-17,PersonX456 are the same as [MetaAttack](https://github.com/FlyingRoastDuck/MetaAttack_AAAI21) described
- Download UnrealPerson from [link](https://github.com/FlyHighest/UnrealPerson)
    - zip unreal_vX.Y and put them to ./data/unrealperson/raw
    - final structure as follows:
<pre>
.
+-- data 
|   +-- unrealperson
|       +-- images
|       +-- meta.json
|       +-- splits.json
|       +-- raw
|           +-- unreal_vX.Y
|               +-- images
</pre>
 - Download RandPerson(all images) from [link](https://github.com/VideoObjectSearch/RandPerson)
    - only download randperson/images/subet and zip it to ./data/randperson/raw/
    - final structure as follows:
 <pre>
.
+-- data 
|   +-- unrealperson
|       +-- images
|       +-- meta.json
|       +-- splits.json
|       +-- raw
|           +-- randperson_subset
|               +-- randperson_subset
</pre>   
## Preparing Attacked re-ID Models
- Download attacked re-ID models from [BaiduYun](https://pan.baidu.com/s/1qH9rj0nJ_ovP8fQTOOzkdw) (Password:zmfm)
- Put models under ./pretrained_models
## Run our Attack Code
- See runAttackMar.sh for more information
## Run our Defense Code
- Preparing perturbation models to ./attackModel, you can pre-download our attacker from [BaiduYun](https://pan.baidu.com/s/1e-BNvD-JoAWTqSuEv27n-A) (Password:oz7a)
- Preparing corresponding pre-trained model from [BaiduYun](https://pan.baidu.com/s/1qH9rj0nJ_ovP8fQTOOzkdw) (Password:zmfm)
- See runDefenseMar.sh for more information
## Evaluate our Defense Models
- Using 'resMeta' to create model, then load defense models 
- You can download our defense models from [BaiduYun](https://pan.baidu.com/s/12ZHPx3lNOP__drjqZ6zO8g) (Password:9p3r)
## Acknowledgments
Our code is based on [MetaAttack](https://github.com/FlyingRoastDuck/MetaAttack_AAAI21), 
if you use our code, please also cite their paper.
```
@inproceedings{yang2021learning,
  title={Learning to Attack Real-World Models for Person Re-identification via Virtual-Guided Meta-Learning},
  author={Yang, Fengxiang and Zhong, Zhun and Liu, Hong and Wang, Zheng and Luo, Zhiming and Li, Shaozi and Sebe, Nicu and Satoh, Shin’ichi},
  booktitle={AAAI},
  volume={35},
  number={4},
  pages={3128--3135},
  year={2021}
}
```










