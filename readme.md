# [Machine Perception](http://www.vvz.ethz.ch/Vorlesungsverzeichnis/lerneinheit.view?lerneinheitId=150922&semkez=2021S&ansicht=LEHRVERANSTALTUNGEN&lang=en) 2021, Course Project

**+++ To find the leonhard submission command, navigate to the end of the [Usage in Leonhard](#usage-in-leonhard) section. +++**

- [Machine Perception 2021, Course Project](#machine-perception-2021-course-project)
    - [Team](#team)
    - [Project Description](#project-description)
    - [Dataset](#dataset)
    - [Method](#method)
        - [Phase 1.](#phase-1)
        - [Phase 2.](#phase-2)
    - [Model](#model)
        - [UXceptionNet](#uxceptionnet)
        - [URegNetY++](#uregnety)
        - [Discriminator](#discriminator)
    - [Augmentations](#augmentations)
        - [Scaling](#scaling)
        - [Rotation](#rotation)
        - [Pixel Intensity Variation](#pixel-intensity-variation)
        - [Hands Flip](#hands-flip)
    - [Usage](#usage)
    - [Requirements](#requirements)
    - [Usage in Leonhard](#usage-in-leonhard)

## Team 
-  **Daniele Chiappalupi**  <br>dchiappal@student.ethz.ch
-  **Pierre Motard** <br>pmotard@student.ethz.ch
-  **Andrea Ziani** <br>aziani@student.ethz.ch

## Project Description
The goal of the project is to create a model able to estimate an amodal segmentation mask for each hand given an RGB image of hands. Below is a sample:

![sample_prediction](/assets/sample.png)

We tackled this task combining a main segmentation network to generate the predictions, a discriminator to make sure that the predictions look like valid hands, and a generator network that makes possible the utilization of the [unpaired images](#dataset).

## Dataset
The data used is a subset of the [InterHand2.6M dataset](https://arxiv.org/abs/2008.09309). The train and validation datasets add up to roughly 68000 128x128 RGB images with custom segmentation masks of left hand and right hand parts. Moreover, we were provided of two additional training sets with roughly 519000 samples each: one contains only RGB images, the other one contains only segmentation annotations. However, they are unpaired, hence they can't be used in a fully-supervised manner.

## Method
The key idea behind our solution is to make use of the unpaired training set to better generalize over unseen data. In order to achieve this, we split the training in two different phases, as shown below.

![sample_prediction](/assets/architecture_and_losses.png)

### Phase 1.
During the first phase, a UXceptionNet segmentation network is trained on the paired training set to produce segmentation masks. Hereby, we will call it G1. In this case, we apply the Cross-Entropy loss function and, as InterHand images are characterized by a heavy class imbalance (i.e. very few pixels represent the fingers and many more pixels depict the background and the areas of the palm and back of the hands), classes are manually weighted to give more importance to those with fewer pixels.

Furthermore, when a hand is heavily occluded, the segmentation network may produce a degenerate mask. To address this problem, we train the discriminator network to classify whether the predicted labels are real or fake. Hereby, we will call it D. We train D in this phase, and we use it in the next phase to provide an adversarial loss for the second segmentation model. In this way, the final segmentation network will be discouraged to generate masks that do not look like valid hands. 

### Phase 2.
After having successfully trained the two aforementioned neural networks, the second phase begin, and here we train the URegNetY++, which will be the final segmentation network. Hereby, we will call it G2. This phase is itself split into two steps:

- *Weights Initialization*: In order to initialize the weights of G2, we train it considering the imperfect labels generated from G1 as ground-truths. This part of training on imperfect masks yields a good weights initialization which helps the network generalizing well. To avoid G2 being influenced by very poorly generated labels, we drop at each epoch half of the images of the batch on which G1 is less confident about its predictions. From tuning the losses, we reached the conclusion that the best performances result from a linear combination of the same Cross-Entropy Loss used in the first phase for G1 and the [Dice Loss](https://arxiv.org/abs/1911.02855) which is well suited for class imbalanced problems.
- *Paired Training*: In this step, we train G2 on the paired training set. Also, as previously mentioned, we use D to help the segmentation network creating predictions as close as possible to the ground-truth distribution. The loss function for this step is a linear combination of Cross-Entropy Loss, Dice Loss and Adversarial Loss.

These steps are repeated twice, for a different amount of epochs. Firstly, the *Weights Initialization* step for more epochs than the *paired training* step. Secondly, the same sequence but with a reverse balance of epochs for each step.

## Model
We will use three different networks in our final architecture. See [Method](#method) to understand how these are combined.

### UXceptionNet
Our first segmentation network, initialized with pre-trained weights on the ImageNet dataset, is a variation of the [U-Net](https://arxiv.org/abs/1505.04597) architecture which uses the [Xception](https://arxiv.org/abs/1610.02357) encoder.

### URegNetY++
The second segmentation network is a [U-Net++](). For this architecture, as the limited time did not give us the chance to apply a more complex encoder, we decided to make use of a [RegNetY32](https://arxiv.org/abs/2003.13678), a good trade-off between complexity and training time. This encoder has  been initialized with weights pre-trained on the ImageNet dataset.

### Discriminator
Lastly, the discriminator model is a simple convolutional network consisting of 5 convolution layers with 4x4 kernel and [64, 128, 256, 512, 1] channels with stride of 2.  Each convolution layer, except for the last, is followed by a batch normalization layer and a [Leaky-ReLU](https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf) layer with a negative slope of 0.2.

We leveraged the great library [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) to create the two segmentation models and retrieve their pre-trained weights.

## Augmentations
Training a deep neural network is highly data-demanding, and the training dataset we have is rather small for the task we want to address. A first step to solve this problem is to use extensive data augmentation to increase the available samples significantly. We applied this technique by performing randomly parameterized transformations on every image of the training set (both paired and unpaired) and the corresponding ground-truth mask if present.

By augmenting the data, the network will have a lower probability of incurring the same image more than once, thus reducing the risk of overfitting. Below are the transformations we used.

### Scaling
The scaling parameter is sampled from a Gaussian distribution with mean $0$ and standard deviation 0.01^2, clipped between -1 and 1. 

### Rotation
The rotation parameter is sampled from a Gaussian distribution with mean $0$ and standard deviation 5^2, clipped between -2 and 2. The rotation is applied with probability 0.6.

### Pixel Intensity Variation
The pixel intensity variation parameters are three random numbers sampled from a uniform distribution between 0.99 and 1.01; each channel of the input image is multiplied pixelwise with the corresponding parameter, and the values are then clipped between 0 and 255.

### Hands Flip
With probability 0.5, we exchange the right and left hands in the input image, performing a horizontal flip.

## Usage

The script that triggers the project is the following:
```bash
python src/train.py [--trainsplit] [--valsplit] [--semisplit] [--input_img_size] 
                    [--load_ckpt] [--final] [--final_splitted] [--init_splitted] 
                    [--eval_every_epoch] [--lr] [--batch_size] [--num_workers] [--save_path] 
                    [--segmentation_model_g1] [--segmentation_model_g2] [--datasets] [--pre] 
                    [--init] [--saved_weights_path_g1] [--segm_loss_w] [--main_loss_w] 
                    [--discriminator_w] [--saved_weights_path_D] [--comet_name] [--use_comet] 
                    [--use_weights] [--use_old_weights] [--loss] [--dryrun]
```
For an explanation of the parameters, check out [config.py](/src/config.py).

## Requirements

The external libraries we used are listed in the [setup](/src/requirements.txt) file.

You can use this file to create `conda` virtual environment with:
```cmd
conda create --name <env> --file requirements.txt
```

## Usage in [Leonhard](https://scicomp.ethz.ch/wiki/Leonhard)

Here is the workflow we followed to train the models inside the Leonhard cluster.

We need to load the required modules including python, CUDA and cuDNN provided on the server:

```cmd
module load gcc/8.2.0 python/3.6.0 cuda/10.1.243 eth_proxy
```

Then we'll create a conda environment in order to install and save the required dependencies. We'll need to install `miniconda` first:

```bash
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh 
```

Now, we will need to link miniconda to our project folder, which we will call ROOT.

```bash
cd ~
mv ~/miniconda3 $STORAGE
ln -s $STORAGE/miniconda3
source $HOME/miniconda3/bin/activate
```

We should now be able to deactivate the default environment, and create the `conda` environment from our `requirements.txt`, and activate it:

```bash
conda deactivate base
conda create --name eth_hand --file src/requirements.txt
conda activate eth_hand
```

It might be a good idea to add this set of commands to the `.bashrc` file in order to not have to run them every time we access to the cluster. To do so, just add the following lines at the end of `~/.bashrc`:

```bash
module load gcc/8.2.0 python/3.6.0 cuda/10.1.243 eth_proxy
source $HOME/miniconda3/bin/activate
conda deactivate base
conda activate eth_hand
```

Finally, we will need to add the data. The model expects the data to be in a folder called `data`in the root of the project folder. The data should be nested in the following way:

```fx
data/
	InterHand2.6M/
		interhand.lmdb/
			data.mdb
			lock.mdb
		segm_train_val_mp.lmdb/
			data.mdb
			lock.mdb
		datalist_stud_test.pt
		datalist_train_complete.pt
		datalist_train_mp.pt
		datalist_val_mp.pt
		path2id_train.pt
		path2id_val.pt
		unpaired_segm_ids.pt
```

We are now all set to run our model.
Let's check that everything went fine: we'll run an interactive GPU environment to see if the everything is right:

```bash
bsub -Is -n 3 -W 1:00 -R "rusage[mem=4096, ngpus_excl_p=1]" bash
```

We'll have to wait some time for the dispatch. Once we are inside, we'll run the following commands to check that we have no errors. 

```bash
python src/utils/sanity_check.py
python src/train.py --dryrun true
```

If everything was correctly set, the full output of this interactive session should be the following:

```rust
(base) [dchiappal@lo-s4-010 ETHand]$ python src/utils/sanity_check.py 
>>> PyTorch version:
>>> 1.6.0
>>> CUDA version:
>>> 10.1
>>> Num GPU detected:
>>> 1
(base) [dchiappal@lo-s4-010 ETHand]$ python src/train.py --dryrun true
{'batch_size': 16,
 'comet_name': 'experiment',
 'data_dir': './data',
 'datasets': 3,
 'discriminator_w': 0.1,
 'dryrun': True,
 'eval_every_epoch': 1,
 'experiment': None,
 'final': 1,
 'final_splitted': 1,
 'init': 1,
 'init_splitted': 1,
 'input_img_shape': (128, 128),
 'input_img_size': 128,
 'load_ckpt': '',
 'loss': 'dice',
 'lr': 0.00025,
 'main_loss_w': 0.65,
 'num_workers': 8,
 'pre': 1,
 'root_dir': '.',
 'save_path': '',
 'saved_weights_path_D': None,
 'saved_weights_path_g1': None,
 'segm_loss_w': 0.35,
 'segmentation_model_g1': 'uxnet',
 'segmentation_model_g2': 'uplusregnet',
 'semisplit': 'mini',
 'trainsplit': 'minitrain',
 'use_comet': False,
 'use_old_weights': True,
 'use_weights': True,
 'valsplit': 'minival'}
Creating train dataset...
Total number of annotations: 1000
Creating train dataset...
Total number of annotations: 1000
Creating val dataset...
Total number of annotations: 1000
Total number of images: 1000
Creating images dataset...
Experiment Key: ab54315f2
  0%|                                                                                       | 0/62 [00:00<?, ?it/s][W TensorIterator.cpp:924] Warning: Mixed memory format inputs detected while calling the operator. The operator will output channels_last tensor even if some of the inputs are not in channels_last format. (function operator())
100%|██████████████████████████████████████████████████████████████████████████████| 62/62 [00:10<00:00,  6.19it/s]
+++++++ Epoch (1): Pre-Training (Training G1 and Discriminator) --> Loss: 295.97859; Avg. Loss: 4.77385; Disc Loss 14.43524, Avg Disc Loss 0.23283;  +++++++
Validation epoch on __train: 100%|█████████████████████████████████████████████████| 62/62 [00:15<00:00,  3.99it/s]
+++++++ miou___train: 0.03325 +++++++
Validation epoch on __val: 100%|███████████████████████████████████████████████████| 62/62 [00:15<00:00,  4.05it/s]
+++++++ miou___val: 0.04124 +++++++
100%|██████████████████████████████████████████████████████████████████████████████| 31/31 [00:08<00:00,  3.60it/s]
+++++++ Epoch (2): Init Training (Training G2 with generated samples from G1 and Unpaired Images) --> Loss: 90.95974; Avg. Loss: 2.93419;  +++++++
Validation epoch on __train: 100%|█████████████████████████████████████████████████| 62/62 [00:16<00:00,  3.71it/s]
+++++++ miou___train: 0.01775 +++++++
Validation epoch on __val: 100%|███████████████████████████████████████████████████| 62/62 [00:16<00:00,  3.76it/s]
+++++++ miou___val: 0.02597 +++++++
100%|██████████████████████████████████████████████████████████████████████████████| 62/62 [00:15<00:00,  4.08it/s]
+++++++ Epoch (3): Final Training (Training G2 on the paired dataset, with the adversarial loss) --> Loss: 186.80528; Avg. Loss: 3.01299; +++++++
Validation epoch on __train: 100%|█████████████████████████████████████████████████| 62/62 [00:16<00:00,  3.66it/s]
+++++++ miou___train: 0.03216 +++++++
Validation epoch on __val: 100%|███████████████████████████████████████████████████| 62/62 [00:16<00:00,  3.73it/s]
+++++++ miou___val: 0.04129 +++++++
Saved model to: logs/ab54315f2/latest.pt
Saved model to: logs/ab54315f2/latest.pt
Finished training. [1/2]
100%|██████████████████████████████████████████████████████████████████████████████| 31/31 [00:08<00:00,  3.67it/s]
+++++++ Epoch (4): Init Training (Training G2 with generated samples from G1 and Unpaired Images) --> Loss: 71.78403; Avg. Loss: 2.31561;  +++++++
Validation epoch on __train: 100%|█████████████████████████████████████████████████| 62/62 [00:16<00:00,  3.67it/s]
+++++++ miou___train: 0.02142 +++++++
Validation epoch on __val: 100%|███████████████████████████████████████████████████| 62/62 [00:16<00:00,  3.69it/s]
+++++++ miou___val: 0.03105 +++++++
100%|██████████████████████████████████████████████████████████████████████████████| 62/62 [00:15<00:00,  4.09it/s]
+++++++ Epoch (5): Final Training (Training G2 on the paired dataset, with the adversarial loss) --> Loss: 156.89308; Avg. Loss: 2.53053; +++++++
Validation epoch on __train: 100%|█████████████████████████████████████████████████| 62/62 [00:16<00:00,  3.66it/s]
+++++++ miou___train: 0.03034 +++++++
Validation epoch on __val: 100%|███████████████████████████████████████████████████| 62/62 [00:16<00:00,  3.76it/s]
+++++++ miou___val: 0.04087 +++++++
Saved model to: logs/ab54315f2/latest.pt
Saved model to: logs/ab54315f2/latest.pt
Finished training. [2/2]
Creating test dataset...
Total number of annotations: 10000
Creating test dataset...
Total number of annotations: 10000
100%|████████████████████████████████████████████████████████████████████████████| 625/625 [00:32<00:00, 18.99it/s]
100%|██████████████████████████████████████████████████████████████████████| 10000/10000 [00:02<00:00, 4344.75it/s]
100%|██████████████████████████████████████████████████████████████████████| 10000/10000 [00:02<00:00, 4143.96it/s]
Done writing test to: logs/ab54315f2/test.lmdb
Done zipping test to: logs/ab54315f2/test.lmdb.tar.gz
```

Finally, we can submit our project to the GPU queue. This is the submission command we used:

```sh
bsub -n 3 -W 80:00 -B -N -o LOG_NAME -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" python src/train.py
```
Let's break down the arguments of the call:
- `-n 3` means that we are requesting 3 CPUs;
- `-W 60:00` means that the job can't last more than 60 hours. This makes it go into the 120h queue of the cluster.
- `-B` instructs LSF to notify you by e-mail when the job begins.
- `-N` instructs LSF to notify you by e-mail when the job begins.
- `-o LOG_FILENAME` means that the output of the job will be stored into the file `LOG_FILENAME`.
- `-R "rusage[mem=4096, ngpus_excl_p=1]"` describes how much memory we request per CPU (4GB) and how many GPUs we ask (1).
- `-R "select[gpu_model0==GeForceRTX2080Ti]"` explicitly requests a RTX2080Ti GPU for the job. We use it to speed up the run.

Check the [usage](#usage) section to check the available configuration options for the train.py script.

Once the job created by the command above is finished, the project will be completed. All the info regarding the location of the submission file, saved model and performances can be found in the `LOG_FILENAME`.