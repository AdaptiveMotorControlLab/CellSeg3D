 # Annotator for cell segmentation
 
## Create a new conda environment
conda create -n naparienv python=3.8 
conda activate naparienv
pip install -r requirements.txt

## Add conda environment in PyCharm

Preferences > Python Interpreter > Add > Conda environment > Existing Environment >
... > miniconda3 (or anaconda) > envs > naparienv > bin > python3.8

## Launch annotator 

```
python annotator.py
```




