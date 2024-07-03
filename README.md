# Online Time-Informed Kinodynamic Motion Planning of Nonlinear Systems
Paper: ****

**Description**
![image](https://github.com/feimeng93/SamplingReachTubeKinoMP/blob/main/asset/about.png)

Fig.1. An illustration of online time-informed kinodynamic motion planning of nonlinear control systems. (a) A Deep Invertible Koopman operator with control U model, DIKU, is trained offline for the nonlinear systems to obtain equivalent linear systems that enable forward and backward dynamics prediction in the lifted space. (b) Our algorithm randomly samples states in the start and goal sets. It then uses the ASKU to bidirectionally propagate the learned linear dynamics and then recover, generating the forward reachable set $\mathcal{X}^f_k$ (independent blue dashed convex hull) and backward reachable tube $\mathcal{X}^b_{−cost:0}$ (serials of green dashed convex hulls) in near real-time. Their intersections constitute the time-informed set (TIS) $\Omega(cost)$ (black convex polygons). Time-informed tree growth is achieved for the off-the-shelf SKMPs by directly sampling in the TIS. The TIS updates according to the cost of the search tree returned after sufficient iterations to help refine the solution (red line).


## Forward and backward dynamics prediction
![image](https://github.com/feimeng93/SamplingReachTubeKinoMP/blob/main/asset/model.png)

Fig. 2. An overview of deep invertible Koopman operator with control (DIKU) neural network model for long-horizon forward and backward dynamics prediction

![image](https://github.com/feimeng93/SamplingReachTubeKinoMP/blob/main/asset/prediction.png)

Fig. 3. Comparison of forward and backward dynamics prediction by our DIKU and the DKU with consistency loss


## Online time-informed set approximation
**Computation times of the Forward/Backward reachable tubes for the example 2D system in [1]**
|  | [Level Set](https://github.com/HJReachability/helperOC) | [Ellipsoidal Toolbox](https://github.com/SystemAnalysisDpt-CMC-MSU/ellipsoids) | RHJB [1] | Ours | Ours (AS) |
| :-----:| :----: | :----: | :----: | :----: | :----: |
| Forward Reachable Tube | 47.46 | 51.13 | 22.17 | **0.03**|**0.16**|
| Backward Reachable Tube| 5.71 | 6.07 | 23.31 |**0.03** | **0.16** |

![image](https://github.com/feimeng93/SamplingReachTubeKinoMP/blob/main/asset/rt.png)

Fig. 4. Comparison results of the example 2D system [1] (left) forward reachable sets and tube and (right) the backward counterparts of TIS computed by the level set toolbox (Ground truth, black), ellipsoidal toolbox (ET, cyan), relaxed HJB equation method [1] (RHJB, purple), our basic sampling-based convex approximation (Ours, red), and our adversarial sampling for over-approximation, ASKU, (Ours(AS), green). Our method provides inner-approximated and tight over-approximated TISs online.

## Online time-informed kinodynamic motion planning

![image](https://github.com/feimeng93/SamplingReachTubeKinoMP/blob/main/asset/render.png)

Fig. 5 Planning solutions (black lines) of the (a) 2D-L, (b) 3D-PNL, (c) CartPole, (d) DampingPendulum, (e) Two-link acrobot, and (f) planar quadrotor systems facing obstacles (blue boxes) given a start (green point) and goal (red pentagram) states. For (c)-(e), both the workspace and state-space trajectories are shown for each method. The orange points are samples. It can be seen that our online time-informed SKMP has a restricted search domain, indicating our ASKU is generalizable and scalable to nonlinear systems and benefits heuristic sampling. The planning efficiency is then improved.



## Requirement
```
pytorch, gym, jupyter, scipy, matplotlib, numpy, cvxpy, dill, tensorboard, pybullet, sklearn
```
## Usage
### Training the DIKU model and evaluation
```
cd Deepkoopman/train/
bash train.sh
```
to generate training trajectory datasets and train the networks (feel free to ask for the dataset used in this paper if you need).

After obtaining the DIKU neural network models, you can use the _Comparison_DIKU_prediction.ipynb_ in folder _Deepkoopman/_ to evaluate the bidirectional prediction performance as shown in Fig. 3.
### Online Time-informed Set Approximation
To evaluate our ASKU method, please 
```
cd ASKU
run Online_TIS.ipynb
```
to reproduce Fig. 4.

## Acknowledgment
[Sampling-based Reachability Analysis: A Random Set Theory Approach with Adversarial Sampling](https://github.com/StanfordASL/UP/)


# BibTex
```
```

# Reference
[1] Tang, Yongxing, Zhanxia Zhu, and Hongwen Zhang. "A Reachability-Based Spatio-Temporal Sampling Strategy for Kinodynamic Motion Planning." IEEE Robotics and Automation Letters 8.1 (2022): 448-455.



