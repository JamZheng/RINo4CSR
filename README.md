# RINo4CSR

## Dataset
The dataset should be organized as the following format. The first column is the userid, followed by the interacted items.

```python
# ./data/dataset_name.txt
user item1 item2 ...
```
## Usage
You can train RINoCSR on Yelp dataset by following command
```bash
python -u main.py --dataset Yelp 
```

## Acknowledgement
The Transformer layer is implemented based on [recbole](https://github.com/RUCAIBox/RecBole).