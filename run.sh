#!/bin/bash

source ~/.virtualenvs/m5-forecasting/bin/activate

ipython --pdb -m m5.train
