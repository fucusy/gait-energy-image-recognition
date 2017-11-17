# gait reconition by gait energy image

## code running environment
language version: python 3.5
os system: Linux or Unix, test on OS X 10.11.5

## python library dependency
* scikit-learn==0.17.1

## algorithm pipeline

do normalization and horizontal alignment to extracted silhouette sequences
which are provided from CASIA Dataset B, 
more detail from the paper `*(2005PAMI)The HumanID Gait Challenge 
Problem_Data Sets, Performance, and Analysis*`


## how to run

1. update the config.py file variable setting, `casia_dataset_b_path`
2. run the main.py script
3. take a look at log file, main.py.log, run `tail -f main.py.log`



## method detail 
read more from paper `*Individual Recognition Using Gait Energy Image*`
