# HowTo100M code

This repo provides code from the HowTo100M paper.
We provide implementation of:
- Our training procedure on HowTo100M for learning a joint text-video embedding
- Our evaluation code on MSR-VTT, YouCook2 and LSMDC for Text-to-Video retrieval
- A pretrain model on HowTo100M
- Feature extraction from raw videos script we used

More information about HowTo100M can be found on the project webpage: https://www.di.ens.fr/willow/research/howto100m/


# Requirements
- Python 3
- PyTorch (>= 1.0)
- gensim


## Video feature extraction

This separate github repo: https://github.com/antoine77340/video_feature_extractor
provides an easy to use script to extract the exact same 2D and 3D CNN features we have extracted in our work.

## Downloading a pretrained model
This will download our pretrained text-video embedding model on HowTo100M.

```
mkdir model
cd model
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/models/howto100m_pt_model.pth
cd ..
```

## Downloading meta-data for evaluation (csv, pre-extracted features for evaluation, word2vec)
This will download all the data needed for evaluation.

```
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/metadata_eval.zip
unzip -d metadata_eval.zip
```

## Downloading meta-data for training on HowTo100M (for the very brave folks :))
This will download all the additional meta data needed for training on HowTo100M.

```
cd data
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/HowTo100M.zip
unzip -d HowTo100M.zip
cd ..
```

## Evaluate the HowTo100M pretrained model on MSR-VTT, YouCook2 and LSMDC

This command will evaluate the off-the-shelf HowTo100M pretrained model on MSR-VTT, YouCook2 and LSMDC.
```
python eval.py --eval_msrvtt=1 --eval_youcook=1 --eval_lsmdc=1 --num_thread_reader=8 --embd_dim=6144 --pretrain_path=model/howto100m_pt_model.pth
```

Note that for MSR-VTT this will evaluate on the same test set as in JSFusion (https://arxiv.org/abs/1808.02559) work
and that YouCook2 will be evaluated on the validation set since no test set is provided.

## Fine-tune the pretrained model on MSR-VTT, YouCook2 and LSMDC

To fine-tune the pretrained model on MSR-VTT, just run this:
```
python train.py --msrvtt=1 --eval_msrvtt=1 --num_thread_reader=4 --batch_size=256 --epochs=50 --n_display=200 --lr_decay=1.0 --embd_dim=6144 --pretrain_path=model/howto100m_pt_model.pth
```

You can also fine-tune on YouCook2 / LSMDC with the parameter --youcook=1 / --lsmdc=1 instead of --msr-vtt=1.

Same thing to monitor evaluation on other dataset with the parameters --eval_msrvtt and --eval_lsmdc.

## Training embedding from scratch on HowTo100M

If you were brave enough to download all the videos and extract 2D and 3D CNN features for them using our provided feature extractor script, we also provide you a way to train the same embedding model on HowTo100M.

First, you will need to extract features for all the HowTo100M videos using our 2D and 3D features extraction script: https://github.com/antoine77340/video_feature_extractor.
The default folder you need to extract the 2d (resp. 3d) features is in 'feature_2d' (resp. 'feature_3d'), but the default folder can be modified by changing the argument --features_path_2D (resp. --features_path_3D). After specifying the root folder for the 2d and 3d features, please use the exact same relative output path name for the features.

Then, modify the training HowTo100M training CSV file (extracted in data/HowTo100M_v1.csv) by adding an additional column 'path', which points to the .npy feature path of each video you have extracted using the provided script.

For example, the modification will look like this:

Original HowTo100M CSV file:
```
video_id,category_1,category_2,rank,task_id
nVbIUDjzWY4,Cars & Other Vehicles,Motorcycles,27,52907
CTPAZ2euJ2Q,Cars & Other Vehicles,Motorcycles,35,109057
...
_97kyZVWVG0,Hobbies and Crafts,Crafts,34,119814
gkjnR3-ZVts,Food and Entertaining,Drinks,6,25938
```

Updated CSV file with features path:

```
video_id,category_1,category_2,rank,task_id,path
nVbIUDjzWY4,Cars & Other Vehicles,Motorcycles,27,52907, path/to/feature/vid_nVbIUDjzWY4.npy
CTPAZ2euJ2Q,Cars & Other Vehicles,Motorcycles,35,109057, path/to/feature/vid_CTPAZ2euJ2Q.npy
...
_97kyZVWVG0,Hobbies and Crafts,Crafts,34,119814, path/to/feature/vid__97kyZVWVG0.npy
gkjnR3-ZVts,Food and Entertaining,Drinks,6,25938, path/to/feature/vid_gkjnR3-ZVts.npy
```


Eventually, this command will train the model and save a checkpoint to the ckpt folder every epochs.

 ```
 python train.py --num_thread_reader=8 --epochs=15 --batch_size=32 --n_pair=64 --n_display=100 --embd_dim=4096 --checkpoint_dir=ckpt --features_path_2D=folder_for_2d_features --features_path_3D=folder_for_3d_features --train_csv=train.csv
 ```
Note that you have to replace "folder_for_2d_features" and "folder_for_3d_features" with the directory used to extract the 2d and 3d video features.

## If you find the code / model useful, please cite our paper
```
@inproceedings{miech19howto100m,
   title={How{T}o100{M}: {L}earning a {T}ext-{V}ideo {E}mbedding by {W}atching {H}undred {M}illion {N}arrated {V}ideo {C}lips},
   author={Miech, Antoine and Zhukov, Dimitri and Alayrac, Jean-Baptiste and Tapaswi, Makarand and Laptev, Ivan and Sivic, Josef},
   booktitle={ICCV},
   year={2019},
}
```
