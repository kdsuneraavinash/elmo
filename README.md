# ELMo Model

## Sources

### Research Papers

[Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf)

[Semi-supervised sequence tagging with bidirectional language models](http://ai2-website.s3.amazonaws.com/publications/semi-supervised-sequence.pdf)

### BiLM Tensorflow implementation Repository

Tensorflow implementation of contextualized word representations from bi-directional language models.This repository supports both training biLMs and using pre-trained models for prediction.

[Link](https://github.com/allenai/bilm-tf)

### AllenNLP

The main website of AllenNLP, an open-source NLP research library built on [PyTorch](https://en.wikipedia.org/wiki/PyTorch). Contains some models and information on ELMo.

[Link](https://allennlp.org/) [About ELMo](https://allennlp.org/elmo) 

### Elmo-Tutorial

A short tutorial on Elmo training (Pre trained, Training on new data, Incremental training) Contains a Jupyter Notebook on how to use ELMo.

[Repo](https://github.com/PrashantRanjan09/Elmo-Tutorial) [Notebook](https://github.com/PrashantRanjan09/Elmo-Tutorial/blob/master/Elmo_tutorial.ipynb)

### State-Of-The-Art Named Entity Recognition With Residual LSTM And ELMo

Article on ELMo and LSTM. Uses Keras.

[Link](https://www.depends-on-the-definition.com/named-entity-recognition-with-residual-lstm-and-elmo/)

## Instructions for training ELMo

### Initializing a `conda`environment

1. download and install `conda`.

2. Activate `conda` in path. (add at the end of `~/.bashrc`)

   ```bash
   export PATH=~/miniconda3/bin:$PATH
   ```

3. Create python 3.6 version environment.

   ```bash
   $ conda create -n ml python=3.6
   ```

4. Activate environment.

   ```bash
   $ source activate ml
   ```

5. Make sure `pip`version is 3.6 (ml).

   ```bash
   $ pip -V
   ```

6. Use [this link](https://cloud.google.com/compute/docs/gpus/add-gpus#install-gpu-driver) to install drivers

7. Install `tensorflow-gpu` and other required libraries using `conda install`.

### Fix weight dumping errors

```bash
$ export CUDA_VISIBLE_DEVICES=1
```

### Training and testing

1. Edit `split.bash` to include train path and result path.

2. Split the train data files.

   ```bash
   $ bash bin/split.bash
   ```

3. Format the data files. (*No need to do this if files are already properly formatted*).

   ```bash
   $ python bin/line.py --src data --dst train --shuffle False
   ```

4. Move some files to a `test` directory. *No Need if not expecting to test.*

5. Run following commands. (Here 50 is the number of files to load at a time)

   ```bash
   $ python bin/create_vocab.py --src train/ --load_files 50
   ```

6. Copy result.

   ```
   Max word length = 25
   Number of tokens = 1462904
   ```

   If forgot, get the number of lines using,

   ```bash
   $ wc -l vocab.txt
   ```

7. Switch to a machine with high-end gpu. Need to transfer `vocab.txt`, `options.json`,  `train/*` and `test/*`. Use `zip` and `unzip` commands and `gsutil` to port the files to the GPU machine.

8. Install `bilm-tf`.

   ```bash
   $ git clone https://github.com/allenai/bilm-tf.git 
   $ python setup.py install
   ```

7.  Make `save`and `log`directories and run this command. (Here 1462904 is the number of lines in the `vocab.txt`)

   ```bash
   $ python bin/train_elmo.py --save_dir save --log_dir log --vocab_file vocab.txt --train_prefix 'train/*' --batch_size 128 --n_gpus 1 --n_epochs 10 --n_train_tokens 1462904
   ```

8. Run this command if need to test perplexity.

   ```bash
   $ python bin/run_test.py --save_dir=save --vocab_file=vocab.txt --test_prefix=test/* --batch_size=128
   ```

9. Dump weights.

   ```bash
   $ python bin/dump_weights.py --save_dir save --out weights.hdf5
   ```

10. Copy `options.json` from `save/` to root directory and change `n_characters` to 262 (From 261)
11. Change file names in query.py and run it.

