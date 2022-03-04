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

### Add widget :
- Add to napari.yaml
- Add to plugin.py
### TODO :
Broken :
- [X] Fix napari_view_simple : replaced deprecated .Gui() with 
.show(), replaced dims.events.axis.connect() with dims.events.current_step.connect()
- [X] maybe remove viewer argument : no, replaced view1 with viewer, changed view_image to add_image
- find better way to pass viewer ?
- [ ] Fix fileread in utils.py : loading only images.tif rn
- [ ] Fix utils.save_masks in utils

Opti :
- [ ] -> Optimize launch time for loader
- Combine widgets from magicgui for dir. management ?
- Port to Qt inst/d of magicgui ? or simply combine file management widgets ?
- Split widgets into loose files to avoid heavy lib import each time ?
- opt imports of libs ?

- [ ] Check min requirements to reduce install time/bloat/compatibility 
- [ ] Add reqs to setup.cfg to avoid pip -r reqs.txt step
- [ ] Rm window popup when launching widgets ?
- [ ] Test if functional with actual dataset/labels
- [ ] Possible improvements ? 


