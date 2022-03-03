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

## Launch annotator 

From folder :
```
pip install -r requirements.txt
cd ..
pip install -e napari-cellseg-annotator
```

### Add widget :
- Add to napari.yaml
- Add to plugin.py
### TODO :
- [ ] Check min requirements to reduce install time/bloat/compatibility 
- [ ] Add reqs to setup.cfg to avoid pip -r step
- [ ] Rm window popup when launching widgets ?
- [ ] split widgets into loose files to avoid heavy lib import each time ?
- [ ] Test if functional with dataset/labels
- [ ] Possible improvements ? 

