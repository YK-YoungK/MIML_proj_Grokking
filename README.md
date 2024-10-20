# Towards Understanding Grokking Phenomenon via Model and Training Efficiency

Final project for Mathematical Introduction to Machine Learning course in Peking University (2023 fall).

## Installation
An optional first step, which will make everything easier:
```bash
conda create -n grok python=3.8
conda activate grok
```
Then, install necessary packages:
```bash
pip install -e .
```

## Experiments
We implemented three network architectures: Transformer, LSTM, and MLP and one special network architecture (MixMLP) for some synthetic experiments. To start training with specific model, first fill in your wandb API in the python script ```./scripts/train-"model".py```:
```python
os.environ["WANDB_API_KEY"] = "Your Key Here"
```
Then start the training program by:
```bash
./scripts/train-"model".py
```
where ```"model"=transformer, lstm, mlp, mixed-mlp```. Also, you can use command line to specify more arguments, such as training percentage, learning rate, etc. For available arguments, you can check in ```add_args()``` in ```training.py```.

## Explanation
We followed prior works to explain grokking phenomenon, verifying on more network architectures and training configs. For more details, you can refer to the report.
