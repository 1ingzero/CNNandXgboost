import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import  _lenet5_backward
import  numpy as np

_X = "/home/zhenxing/DATA/CYP DATASET/Pubchem410/CYP1A2_txt/CYP1A2_PaDEL12_pcfp_train_x.txt"
_Y = "/home/zhenxing/DATA/CYP DATASET/Pubchem410/CYP1A2_txt/CYP1A2_PaDEL12_pcfp_train_y.txt"
DATA_X = np.loadtxt(_X, dtype=np.float32)
DATA_Y = np.loadtxt(_Y, dtype=np.float32)
DATA_Y_OH = _lenet5_backward.trans_OH(DATA_Y)
T_X = "/home/zhenxing/DATA/CYP DATASET/Pubchem410/CYP1A2_txt/CYP1A2_PaDEL12_pcfp_test_x.txt"
T_Y = "/home/zhenxing/DATA/CYP DATASET/Pubchem410/CYP1A2_txt/CYP1A2_PaDEL12_pcfp_test_y.txt"
TDATA_X = np.loadtxt(T_X, dtype=np.float32)
TDATA_Y = np.loadtxt(T_Y, dtype=np.float32)
TDATA_Y_OH = _lenet5_backward.trans_OH(TDATA_Y)
_lenet5_backward.backward(DATA_X, DATA_Y_OH,TDATA_X,TDATA_Y_OH)

