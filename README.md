# river-profile-lidar

conda create -n river-profile-lidar python=3.12.4
conda activate river-profile-lidar

pip install -r requirements-dev.txt
pre-commit install

Make sure that the folder specifiy in the config files exists

python -m cross_section_points river=dartmouth
python -m trapezoid_bathymetry river=dartmouth
python -m hab river=dartmouth
python -m water_speed river=dartmouth
python -m d84 river=dartmouth
python -m iqh_surfacique river=dartmouth
python -m iqh_linear river=dartmouth
python -m iqh_transects river=dartmouth
