#!/bin/bash

python train.py --fold 0 --model_name logistic_regression
python train.py --fold 1 --model_name logistic_regression
python train.py --fold 2 --model_name logistic_regression
python train.py --fold 3 --model_name logistic_regression
python train.py --fold 4 --model_name logistic_regression