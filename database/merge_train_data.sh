#!/bin/bash
rm -f train_data.csv
for i in `seq -w 01 05`
do 
    unzip -o train_data_${i}.zip
    cat train_data_${i}.csv >> train_data.csv
    rm -f train_data_${i}.csv
done

