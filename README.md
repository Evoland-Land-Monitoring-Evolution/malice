# Multimodal Multiview Self-supervised learning for Satellite Image Time Series (SITS)

This code is designed for self-supervised spatio-temporal representation learning of multi-modal Satellite Image Time Series (SITS) Sentinel-1 and Sentinel-2.
The code also allows to add auxiliary data to each SITS : AGERA5 weather data and digital elevation model (DEM).

The two models are respectively named MALICE and MALICE Aux.

[//]: # (# Easy import of ALISE )

[//]: # (To solely use ALISE for inference &#40;no training&#41;, a onnx format of the model is available. The model is available at : [https://zenodo.org/records/10838982]&#40;https://zenodo.org/records/10838982&#41;)

[//]: # (To use ALISE : )

[//]: # (- Irregular SITS of spatial dimension &#40;1,t,c,h,w&#41;, with t which can vary, c=10 &#40;10 S2 bands&#41;. The pre-training was conducted with h=w=64, but to use the model on downstream tasks it is not required. )

[//]: # (- A batch of irregular SITS. To create a batch a SITS of varying length temporal padding may be applied. In this case )

[//]: # (the padding mask should be provided to ALISE )

[//]: # (- Rescale the SITS given the stats available at : https://zenodo.org/records/10838982 the SITS need to be rescaled before being processed by ALISE.)

[//]: # (The notebook ./notebook/alise_feature_extraction.ipynb gives a simple, user-friendly example on how to use this pre-trained SITS encoder.)

[//]: # ()
[//]: # (## requirements for easy import )

[//]: # (- onnx [https://pypi.org/project/onnx/]&#40;https://pypi.org/project/onnx/&#41;)

[//]: # (- onnxruntime [https://pypi.org/project/onnxruntime/]&#40;https://pypi.org/project/onnxruntime/&#41;)

[//]: # (- modcix dataset [https://src.koda.cnrs.fr/iris.dumeur/modcix]&#40;https://src.koda.cnrs.fr/iris.dumeur/modcix&#41;)

[//]: # (- openeommdc `https://src.koda.cnrs.fr/iris.dumeur/openeo_datasets`)
# Requirements
To run the code available in the repo for fine-tuning or pre-training a model from the scratch the following repo need to be downloaded :

[//]: # (- Install torchmuntan : https://gitlab.cesbio.omp.eu/activites-ia/torchmuntan)
- Install openeo_mmdc: https://src.koda.cnrs.fr/iris.dumeur/openeo_datasets
- Install mtan_s1s2_classif: https://src.koda.cnrs.fr/mmdc/mtan_s1s2_classif
- Install pastis_eo_dataset : https://src.koda.cnrs.fr/iris.dumeur/pastis_eo_dataset
- Install mt_ubarn_res : [https://src.koda.cnrs.fr/iris.dumeur/mt_ubarn_results](https://src.koda.cnrs.fr/iris.dumeur/mt_ubarn_results/-/tree/master?ref_type=heads)
- Install mt_ssl : https://src.koda.cnrs.fr/iris.dumeur/alise
- Modify and run `./create-conda-env.sh`

- This code relies on Pytorch lightning 2.0 and Hydra. Hydra is used as a great and flexible argument parser. 

# Model pretraining

To pretrain MALICE model, please, modify files `config/pretrain_proj.yaml`, `config/module/alise_mm_proj.yaml`, `config/datamodule/mask_mm.yaml` and `config/dataset/mask_mm.yaml`.

To pretrain MALICE Aux model, please, modify files `config/pretrain_proj.yaml` and `config/module/alise_mm_proj.yaml`, `config/datamodule/mask_mm_aux.yaml` and `config/dataset/mask_mm_aux.yaml`.


# Inference

Pretrained models are available in folder `pretrained_models`. 
For the inference example, see `pretrained_models/inference_malice.py` and `pretrained_models/inference_malice_aux.py`.