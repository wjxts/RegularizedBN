
We present the code and instructions to reproduce our NeurIPS 2022 paper  **"Understanding the Failure of Batch Normalization for Transformers in NLP"** on neural machine translation experiments.   
For other tasks, you can easily modify the normalization module in [language modeling](https://github.com/szhangtju/The-compression-of-Transformer), [named entity recognition](https://github.com/fastnlp/TENER), [text classification](https://github.com/declare-lab/identifiable-transformers) to reproduce the corresponding results. For the reason of license, we do not include them here. We are still appending new features. 

The codes are based on [fairseq](https://github.com/pytorch/fairseq) (v0.9.0)

BN/RBN module is located at: fairseq\modules\norm\mask_batchnorm3d.py

# Reproduction

Install [PyTorch](http://pytorch.org/) (we use Python=3.6 and PyTorch=1.7.1, higher version of python and PyTorch should also work)
```bash
conda create -n rbn python=3.6
conda activate rbn
conda install pytorch==1.7.1 torchvision torchaudio cudatoolkit=11.0 -c pytorch (or pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html)
```
Install fairseq by:  

```bash
cd RegularizedBN
pip install --editable ./
```
Install other requirements
```bash
pip install -r requirements.txt
```
# IWSLT14 De-En

Download the data from [google drive](https://drive.google.com/file/d/1p8MxfqRPe_tzVwyiUmIsq-q6AG3jF22V/view?usp=sharing) and extract it in data-bin.  You can also download it from  [Baidu Netdisk](https://pan.baidu.com/s/1DQgjBGuorZ0QqKKW0YcvvA?pwd=znde).  
```bash
cd data-bin
unzip iwslt14.tokenized.de-en.zip
cd ..
```
Training the model (8GB GPU memory is enough)  

```bash
chmod +x ./iwslt14_bash/train-iwslt14-pre-max-epoch.sh ./iwslt14_bash/train-iwslt14-post-max-epoch.sh 
```


```bash
For Pre-Norm Transformer:  
BN: 
CUDA_VISIBLE_DEVICES=0 ./iwslt14_bash/train-iwslt14-pre-max-epoch.sh batch_1_1
RBN: 
CUDA_VISIBLE_DEVICES=1 ./iwslt14_bash/train-iwslt14-pre-max-epoch.sh batch_diff_0.1_0.01  
LN: 
CUDA_VISIBLE_DEVICES=2 ./iwslt14_bash/train-iwslt14-pre-max-epoch.sh layer_1

```

```bash
For Post-Norm Transformer:  
BN: 
CUDA_VISIBLE_DEVICES=0 ./iwslt14_bash/train-iwslt14-post-max-epoch.sh batch_1_1
RBN: 
CUDA_VISIBLE_DEVICES=1 ./iwslt14_bash/train-iwslt14-post-max-epoch.sh batch_diff_60_0
LN: 
CUDA_VISIBLE_DEVICES=2 ./iwslt14_bash/train-iwslt14-post-max-epoch.sh layer_1
 
```

