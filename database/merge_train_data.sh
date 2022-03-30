#!/bin/bash
# if train_data.csv does not exist, loop through each zip file
if [[ ! -f 'train_data.csv' ]]
then
    for i in `seq -w 01 05`
    do  
        unzip -o train_data_${i}.zip                  # unzip each part
        cat train_data_${i}.csv >> train_data.csv     # concatenate each part into train_data.csv
        rm -f train_data_${i}.csv                     # remove intermediate csv files
        rm -f train_data_${i}.zip                     # remove zip files
    done
fi  
