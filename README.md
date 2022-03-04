 # Annotator for cell segmentation : NPE2 test
 
## Create a new conda environment
```
conda create -n napari-cellseg python=3.8 
conda activate napari-cellseg
pip install -r requirements.txt
```

## Add conda environment in PyCharm

Preferences > Python Interpreter > Add > Conda environment > Existing Environment >
... > miniconda3 (or anaconda) > envs > naparienv > bin > python3.8

## Install & launch annotator 

From folder :
```
pip install -r requirements.txt
cd ..
pip install -e napari-cellseg-annotator
napari
```

- [ ] Fix utils.save_masks in utils


