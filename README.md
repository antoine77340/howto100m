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
python train.py --msrvtt=1 --eval_msrvtt=1 --num_thread_reader=8 --batch_size=256 --epochs=50 --n_display=200 --lr_decay=1.0 --embd_dim=6144 --pretrain_path=model/howto100m_pt_model.pth
```

You can also fine-tune on YouCook2 / LSMDC with the parameter --youcook=1 / --lsmdc=1 instead of --msr-vtt=1.

Same thing to monitor evaluation with the parameters --eval_msrvtt and --eval_lsmdc.

## Training embedding from scratch on HowTo100M

If you were brave enough to download all the videos and extract 2D and 3D CNN features for them using our provided feature extractor script, we also provide you a way to train the same embedding model on HowTo100M.

First you need to download the csv file of HowTo100M.

```
blabla
```

Second, you will need to extract features for all the HowTo100M videos using our 2D and 3D features extraction script: https://github.com/antoine77340/video_feature_extractor

After that you will create a new csv file, let say train.csv, in which you will create
a new column 'path' providing the feature path for each video.

For instance, the original training csv file becomes:

```
...
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
