To run the code 

## Training
- Import all of the py files as a dataset on kaggle, then run through the cells of [runner.ipynb](./runner.ipynb) one by one to setup the kaggle environment and start the training
- Configuration parameters can be changed in [config.py](./config.py) before uploading as dataset or use get_config to change the parameters (see runner.py)


## Test/Inference

- Add dataset splits, tokenizers and model .pt files created during training as dataset to kaggle and run [test_inference](./test_inference.ipynb) cell by cell to run inference using different decoding strategies 


