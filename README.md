# LGC-Plus

Further development and elaboration on the impact of multi-modality for confidence estimation, inspired by [LGC](https://openaccess.thecvf.com/content_ECCV_2018/papers/Fabio_Tosi_Beyond_local_reasoning_ECCV_2018_paper.pdf).

Proposed by [Konstantin Heinrich](http://www.linkedin.com/in/konstantin-heinrich) and [Max Mehltretter](http://mehltretter.net/). 

Our paper can soon be found in the [ISPRS archives](https://www.isprs.org/publications/archives.aspx). Please refer to our paper, when using the code.

## Prerequisites
This code was tested with Python 3.7.7, Tensorflow 2.2.0, CUDA 10.1 on the [cluster system](https://www.luis.uni-hannover.de/en/services/computing/scientific-computing/technical-specifications-of-clusters/) at the Leibniz University of Hannover, Germany, under CentOs 7.6. Additionally, for small tests with an Nvidia Geforce 1060 gtx on Windows 10 was used. 

The following data hierarchy is used:

```
LGC-Plus
├── model
├── results
│   ├── Training
│   ├── Testing
├── data
│   ├── dataset
|   |   ├──  images
|   |   |   ├──  left
|   |   |   ├──  right
|   |   ├──  disp_gt
|   |   ├──  cost_volume
|   |   ├──  est_disp
```

## Training
Several arguments for training and testing are available.

The _--module_name_ argument depicts the architecture:
  * [CVA](https://www.sciencedirect.com/science/article/abs/pii/S0924271620303026)
  * [LFN](http://www.arts-pi.org.tn/rfmi2017/papers/10_CameraReadySubmission_llncs2e%20(3).pdf)
  * [ConfNet](https://openaccess.thecvf.com/content_ECCV_2018/papers/Fabio_Tosi_Beyond_local_reasoning_ECCV_2018_paper.pdf)
  * LGC

General prompt structure:
```shell     
python ./model/Train-module.py --module_name [module] --network_name [name] --epoch 14 --amount_training_data 20 --amount_validation_data 2 --batch_size 128 --data_set KITTI-12
```
The following prompt outline a possible setup of arguments to train the local network CVA.
```shell     
python ./model/Train-module.py --module_name CVA --network_name CVA_GITHUB --epoch 14 --amount_training_data 20 --amount_validation_data 2 --batch_size 128 --data_set KITTI-12
```

For the global network ConfNet additional arguments are applicable. In this example the model is trained on pre-computed MC-CNN-based disparity maps from KITTI 2015, with an additional late fused warped difference input with batch normalisation (BN) in the decoder:
```shell     
python ./model/Train-module.py --module_name ConfNet --network_name ConfNet_GITHUB --epoch 1600 --amount_training_data 20 --amount_validation_data 2 --batch_size 1 --data_set KITTI-15 --sm_method MC_CNN --use_warp LF use_BN True
```

Finally, the fusion module requires the confidence maps of the local and global branches, acquired by testing.
```shell     
python ./model/Train-module.py --module_name LGC --network_name LGC_GITHUB --epoch 14 --amount_training_data 20 --amount_validation_data 2 --batch_size 64 --data_set KITTI-15 --LGC_local_input Confmap_CVA --LGC_global_input Confmap_ConfNet
```

## Testing
Testing follows the same prompt ideology as training:
```shell     
python ./model/Test-module.py --module_name [module] --model_dir [model weights]  --data_set [] 
```

Confidence maps via ConfNet are computed with the following prompt, in this case on the Middlebury-v3 dataset, using the weights from ConfNet_GITHUB
```shell     
python ./model/Test-module.py --module_name ConfNet --model_dir ConfNet_GITHUB --data_set Middlebury-v3
```

For the fusion module confidence maps from the networks used for training are needed:
```shell     
python ./model/Test-module.py --module_name LGC --LGC_local_input Confmap_CVA --LGC_global_input Confmap_ConfNet --data_set KITTI-15 --image_amount 100
```

