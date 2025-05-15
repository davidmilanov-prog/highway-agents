# Highway Agents

Compare a Proximal Policy Optimization (PPO) driver with a Spiking Neural Network surrogate for autonomous control on a 2‑D racetrack.

--

## Quick Start
```bash
# clone and enter the repo
git clone https://github.com/davidmilanov-prog/highway-agents.git
cd highway-agents

# (optional) create a virtual environment
python -m venv venv && source venv/bin/activate

# install deps
pip install -r requirements.txt

# run the PPO policy (TRAIN flag inside script)
python rl_creator.py  

# evaluate the Spiking NN
python snn_eval.py
