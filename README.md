So to get started:
VS code

In anaconda either create environment or use the global one
Then change python interpreter in VS Code to anaconda one

python3 -m pip install gymnasium

sudo apt-get install swig

python3 -m pip install gymnasium[box2d]

pip3 install gymnasium[classic-control]`

HAL-CGP
Clone Git repo and go to cloned dir
Change numpy to 1.23.5 in requirements.txt
sudo pip install -e .

on errors like: https://stackoverflow.com/questions/71010343/cannot-load-swrast-and-iris-drivers-in-fedora-35/72200748#72200748
libGL error: failed to load driver: iris
libGL error: MESA-LOADER: failed to open iris: /usr/lib/dri/iris_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
libGL error: failed to load driver: iris
libGL error: MESA-LOADER: failed to open swrast: /usr/lib/dri/swrast_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
libGL error: failed to load driver: swrast

use 
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6



Main questions:
What is to_func() and why do I need it?

Feedback:

In https://happy-algorithms-league.github.io/hal-cgp/basic_usage.html
population_params = {"n_parents": 10, "mutation_rate": 0.5, "seed": 8188211}
mutation_rate is not part of population_params but instead part of ea_params

seed goes into reset function now as a parameter

poetry pass: evd-acrobot
