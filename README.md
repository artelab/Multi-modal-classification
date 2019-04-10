### Multi-modal classification

This code is the implementation of the approach described in: 

I. Gallo, A. Calefati, S. Nawaz and M.K. Janjua, 
"Image and Encoded Text Fusion for Multi-Modal Classification",
presented at 
2018 International Conference on Digital Image Computing: Techniques and Applications (DICTA), Canberra, Australia, 2018.

If you use this code entirely or partially, please cite our
[paper](http://artelab.dista.uninsubria.it/res/research/bibtex/2018/2018_gallo_dicta.txt).


#### Pre-requisites

In the requirements.txt there are dependencies required by the project in order to work.
However, Tensorflow packages has not been added because you can use either the GPU version
or the standard one.


##### Install Tensorflow

To install Tensorflow, follow instructions at: https://www.tensorflow.org/install


##### Install dependencies

Install dependencies with:
 
`pip install -r requirements.txt`


#### How to use

1. Launch the bash script "train-on-ferramenta.sh"

`bash train-on-ferramenta.sh`

It first download the tar.gz files of the dataset, then extracts it and, finally, launches
the training process.

2. When the process it's over, launch "extract-dataset-ferramenta.sh" script

`bash extract-dataset-ferramenta.sh`

It processes the original dataset making a copy of it, with the images that contain 
encoding of text information on top.

3. With the dataset obtained, you can train a simple CNN for image classification exploiting
the advantages of this approach.


#### Custom parameters

If you want to change the value of parameters used, you can modify values 
contained in **training_parameters.csv** and **extraction_parameters.csv**.

In these 2 files there are also references of where to find the dataset to load and its format.

If you want to run our code on your own multi-modal dataset (containing images and text for each sample),
please check the format of **train.csv** and **val.csv**.


#### Help

If you are in troubles running the code, please contact us:

- Ignazio Gallo: *i.gallo@uninsubria.it*
- Alessandro Calefati: *a.calefati@uninsubria.it* 



