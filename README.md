# river-profile-lidar

1. Create a new conda environnement

```
conda create -n river-profile-lidar python=3.12.4
conda activate river-profile-lidar
```
2. Install requirements
If only running the code
```
pip install -r requirements.txt
```
If working as a dev to commit changes 

```
pip install -r requirements-dev.txt
pre-commit install
```
3. Run the code

Make sure that the folder specifiy in the config files exists
```
python -m cross_section_points river=dartmouth
python -m trapezoid_bathymetry river=dartmouth
python -m hab river=dartmouth
python -m water_speed river=dartmouth
python -m d84 river=dartmouth
python -m iqhp_surfacique river=dartmouth
python -m iqhp_linear river=dartmouth
python -m iqhp_transects river=dartmouth
python -m uphp_table river=dartmouth
```
