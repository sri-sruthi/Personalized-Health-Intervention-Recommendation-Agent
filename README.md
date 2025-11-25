# Health Intervention Reinforcement Learning Agent  
### Personalized Health Recommendations using LSTM Forecasting, PPO, SAC, and Hybrid Reinforcement Learning

## 1. Introduction

This repository implements a personalized health intervention system designed to generate user-specific recommendations based on wearable sensor data. The system integrates sequence modeling (LSTM), reinforcement learning (PPO and SAC), and a hybrid decision-making mechanism to provide context-aware behavioural interventions aimed at improving key health indicators such as sleep efficiency, activity levels, and daily movement patterns.

The project includes a modular codebase, a custom Gymnasium environment, reproducible training pipelines, and an interactive Streamlit interface for demonstration and evaluation.

---

## 2. Key Capabilities

### 2.1 Predictive Modelling  
A Long Short-Term Memory (LSTM) neural network predicts next-day sleep efficiency using a seven-day window of engineered health features. The model incorporates:  
- Multivariate feature sequences  
- Batch normalization and dropout  
- Early stopping and learning rate scheduling  
- A production-ready training configuration  

### 2.2 Reinforcement Learning Agents  
The system trains and evaluates three RL approaches:

#### PPO (Proximal Policy Optimization)
- Discrete action space consisting of four intervention categories  
- Stable policy-gradient optimization  
- Suitable for categorical behavioural recommendations  

#### SAC (Soft Actor-Critic)
- Continuous action space representing intervention intensities  
- Entropy-regularized objective for exploration stability  
- Suited for fine-grained behavioural adjustments  

#### Hybrid PPO + SAC Model
- A gating mechanism selects between PPO and SAC policies dynamically  
- Short rollouts evaluate which agent performs better per state  
- Produces an adaptive and robust unified policy  

### 2.3 Custom Health Environment  
A purpose-built Gymnasium environment captures:  
- Step count, active minutes, sedentary behaviour  
- Sleep efficiency and deep sleep ratios  
- Heart-rate statistics and variability  
- Rolling-window features such as step consistency and sleep debt  
- Reward shaping based on metric improvements and behavioural expectations  

### 2.4 Interactive Dashboard  
A Streamlit dashboard provides:  
- Data loading and preprocessing  
- LSTM model training and inspection  
- PPO, SAC, and Hybrid RL training  
- Algorithm performance comparison  
- Real-time intervention recommendation  

---

## 3. Repository Structure

```
github_release/
│
├── src/
│   ├── health_agent_core.py
│   ├── agent_production.py
│   ├── main_demo.py
│   ├── run_production.py
│   ├── streamlit_dashboard.py
│   ├── training/
│   │   ├── train_lstm.py
│   │   ├── train_rl.py
│   │   └── compare_algorithms.py
│   └── utils/
│       ├── data_processor.py
│       ├── environment.py
│       ├── app.py
│       └── check_env.py
│
├── app/
│   └── production_ui.py
│
├── data/
│   ├── dailyActivity_merged.csv
│   ├── heartrate_seconds_merged.csv
│   └── minuteSleep_merged.csv
│
├── models/
│   └── .gitkeep
│
├── notebooks/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 4. Installation

### 4.1 Clone the Repository
```bash
git clone https://github.com/sri-sruthi/Personalized-Health-Intervention-Recommendation-Agent.git
cd Personalized-Health-Intervention-Recommendation-Agent
```

### 4.2 Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 4.3 Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 5. Usage

### 5.1 Launch the Streamlit Dashboard (Recommended)
```bash
streamlit run app/production_ui.py
```
This interface supports data processing, model training, algorithm comparison, and real-time intervention recommendations.

---

### 5.2 Train the LSTM Model (Programmatic Usage)
```python
from training.train_lstm import ProductionSleepPredictor

predictor = ProductionSleepPredictor()
predictor.train_production(epochs=100)
```

---

### 5.3 Train PPO and SAC Agents
```python
from training.train_rl import FastRLTrainer

trainer = FastRLTrainer()

ppo_model, ppo_env = trainer.train_ppo_agent()
sac_model, sac_env = trainer.train_sac_agent()
```

---

### 5.4 Train the Hybrid Reinforcement Learning Agent
```python
from training.train_rl import HybridAgent, FastRLTrainer

trainer = FastRLTrainer()

hybrid_agent = HybridAgent(
    ppo_path="models/ppo_health_agent",
    sac_path="models/sac_health_agent",
    env=trainer.env
)
```

---

### 5.5 Evaluate All Agents
```python
from training.compare_algorithms import AlgorithmComparator

comparator = AlgorithmComparator()
results = comparator.run_comparison(num_episodes=50)
comparator.plot_comparison(results)
```

---

## 6. Engineered Features

The system constructs a comprehensive health-feature vector including:

- Total daily steps  
- Total active minutes  
- Sedentary minutes  
- Activity–sedentary ratio  
- Rolling seven-day step consistency  
- Sleep efficiency  
- Deep sleep ratio  
- Heart-rate mean, standard deviation, and variability  
- Synthetic sleep-debt estimate  
- Additional physiological and behavioural metrics  

All features undergo normalization and preprocessing through a modular pipeline.

---

## 7. Reinforcement Learning Formulation

### 7.1 PPO  
- Discrete action space representing four intervention types  
- Generalized Advantage Estimation (GAE)  
- MLP-based policy with shared latent layers  

### 7.2 SAC  
- Continuous action space for intervention intensities  
- Entropy regularization for stable exploration  
- Actor–critic architecture with soft updates  

### 7.3 Hybrid PPO + SAC  
- Combines discrete and continuous control  
- Gating network selects the more effective agent per state  
- Improves policy robustness across heterogeneous health states  

---

## 8. Results Summary

Experimental evaluations show:

- PPO performs reliably for discrete behavioural choices  
- SAC excels when fine-grained intervention intensity matters  
- The hybrid model achieves the highest mean reward and stability  
- Overall, the hybrid agent produces the most personalized and context-aware recommendations  

---

## 9. License

This project is released under the MIT License.  
See the `LICENSE` file for details.

---

## 10. Author

Sri Sruthi Manikka Nagasamy  
Integrated MSc Data Science  
Personalized Health Intervention Recommendation Agent Project
Amrita Vishwa Vidyapeetham Coimbatore

---

## 11. Learning Outcomes

The development of this system demonstrates the following competencies:

- Designing and training LSTM models for multivariate time-series forecasting  
- Building a custom Gymnasium reinforcement learning environment for wearable-data simulation  
- Implementing PPO (discrete control) and SAC (continuous control) using Stable-Baselines3  
- Developing a hybrid RL mechanism that combines discrete and continuous action policies  
- Reward shaping, trajectory evaluation, and RL performance analysis  
- Integrating predictive modelling and reinforcement learning into a unified personalized recommendation workflow  
- Constructing a modular, research-grade codebase structured for maintainability  
- Implementing an interactive Streamlit interface for experimentation and demonstration  
- Executing robust preprocessing pipelines for wearable sensor datasets  
- Performing comparative evaluation of RL algorithms using standardized metrics and visualizations
