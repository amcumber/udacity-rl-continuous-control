[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


[//]: # (Link References)

[link1]: https://github.com/udacity/deep-reinforcement-learning/blob/master/p2_continuous-control/README.md "Udacity-README"

# Project 2: Continuous Control

Author: Aaron McUmber

Date: 2021-09-12

### Introduction

The project demonstrates an agent implementeing Deep Deterministic Policy
Gradient (DDPG) to direct and navigate either

* Option 1: one agent or
* Option 2: twenty agents

within a Unity environment to control a double-jointed arm into target
locations. Each arm is rewarded `+0.1` for each step thtat the agent's hand is
within the goal location. The goal of the agent(s) isto maintain it's position
within the target location for as many time steps as possible.

Success is achiveved when either one agent in [Option 1] reaches an average
score of `+30.0` over 100 consecutive episodes or in [Option 2] if the average
of the twenty agents achieve an average score of `+30.0` over 100 consecutive
episodes.


## Getting Started

* Download the supporting modules:
1. Unity ml-agents version 0.4.0 module along with it's required dependencies: 
   (here)[https://github.com/Unity-Technologies/ml-agents]

2. A built version of the Unity environment provided by the 
   (problem statement repo)[https://github.com/udacity/deep-reinforcement-learning/blob/master/p2_continuous-control/README.md]
   as a direct executable for the target environment.- *Note: be
   sure to select the correct environment for your machine*

3. Install the requirements found in `requirements.txt`

``` bash
pip install -r requirements.txt
```
4. Launch a Jupyter notebook and Open `Continuous_Control.ipynb`:

``` bash 
jupyter notebook
```

5. Follow the Notebook - and refer to `report.html` for additional discriptions
