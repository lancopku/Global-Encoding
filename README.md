# Global-Encoding
This is the code for our paper *Global Encoding for Abstractive Summarization*, https://arxiv.org/abs/1805.03989

***********************************************************

## Requirements
* Ubuntu 16.0.4
* Python 3.5
* Pytorch 0.4.1 (updated)
* pyrouge

In order to use pyrouge, set rouge path with the line below:
```
pyrouge_set_rouge_path RELEASE-1.5.5/
```
It seems that some user have met problems with pyrouge, so I have updated the script, and users can put the directory "RELEASE-1.5.5" in your home directory and set rouge path to it  (or run the command "chmod 777 RELEASE-1.5.5" for the permission).
**************************************************************

## Preprocessing
```
python3 preprocess.py -load_data path_to_data -save_data path_to_store_data 
```
Remember to put the data into a folder and name them *train.src*, *train.tgt*, *valid.src*, *valid.tgt*, *test.src* and *test.tgt*, and make a new folder inside called *data*

***************************************************************

## Training
```
python3 train.py -log log_name -config config_yaml -gpus id
```
Create your own yaml file for hyperparameter setting.

****************************************************************

## Evaluation
```
python3 train.py -log log_name -config config_yaml -gpus id -restore checkpoint -mode eval
```

*******************************************************************

# Citation
If you use this code for your research, please kindly cite our paper:
```
@inproceedings{globalencoding,
  title     = {Global Encoding for Abstractive Summarization},
  author    = {Junyang Lin and Xu Sun and Shuming Ma and Qi Su},
  booktitle = {{ACL} 2018},
  year      = {2018}
}
```
