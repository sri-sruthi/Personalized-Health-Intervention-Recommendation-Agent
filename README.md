# Health Intervention Reinforcement Learning Agent  
### Personalized Health Recommendations using LSTM Forecasting, PPO, SAC, and Hybrid Reinforcement Learning

## 1. Introduction

This repository contains the implementation of a personalized health intervention system designed to generate user-specific recommendations based on wearable sensor data. The system integrates sequence modeling (LSTM), reinforcement learning (PPO, SAC), and a hybrid decision-making mechanism to provide context-aware behavioural interventions that aim to improve key health indicators such as sleep efficiency, activity levels, and daily movement patterns.

The project includes a modular codebase, a custom Gymnasium environment, reproducible training pipelines, and an interactive Streamlit interface for demonstration and evaluation.

---

## 2. Key Capabilities

### 2.1 Predictive Modelling
A Long Short-Term Memory (LSTM) neural network predicts next-day sleep efficiency using a seven-day window of engineered health features. The model incorporates:
- Multi-feature input sequences  
- Batch normalization and dropout  
- Early stopping and learning-rate scheduling  
- High-performance training configurations  

### 2.2 Reinforcement Learning Agents
The system trains and evaluates three RL approaches:

#### PPO (Proximal Policy Optimization)
- Discrete action space (four health interventions)  
- Stable and sample-efficient  
- Suitable for categorical behavioural recommendations  

#### SAC (Soft Actor-Critic)
- Continuous action space (intervention intensities)  
- Entropy-regularized objective  
- Captures fine-grained behavioural adjustments  

#### Hybrid PPO + SAC Model
- A gating network selects either PPO or SAC per state  
- Short rollouts compute agent-specific performance  
- Produces a unified and adaptive policy  

### 2.3 Custom Health Environment
The project includes a custom Gymnasium environment that models:
- Daily steps, active minutes, sedentary time  
- Sleep efficiency and deep-sleep ratio  
- Heart-rate variability, HR mean, HR standard deviation  
- Rolling features such as consistency and sleep debt  
- Reward shaping based on improvement across metrics  

### 2.4 Interactive Dashboard
A Streamlit dashboard enables:
- Data loading and preprocessing  
- LSTM model training and evaluation  
- PPO, SAC, and Hybrid RL training  
- Performance comparison  
- Real-time intervention recommendations  

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

### 5.1 Launch the Streamlit Dashboard
```bash
streamlit run app/production_ui.py
```

This interface supports model training, algorithm comparison, and interactive demonstration of recommendations.

### 5.2 Train the LSTM Model
```python
from training.train_lstm import train_lstm
train_lstm()
```

### 5.3 Train PPO and SAC Agents
```python
from training.train_rl import train_ppo, train_sac

ppo_model = train_ppo()
sac_model = train_sac()
```

### 5.4 Train the Hybrid RL Agent
```python
from training.compare_algorithms import build_hybrid
gating_network = build_hybrid()
```

### 5.5 Evaluate All Agents
```python
from training.compare_algorithms import evaluate_all

results = evaluate_all()
print(results)
```

---

## 6. Engineered Features

The system uses a rich set of input features:

- Total daily steps  
- Total active minutes  
- Sedentary minutes  
- Activity-to-sedentary ratio  
- Step consistency (rolling 7-day standard deviation)  
- Sleep efficiency  
- Deep sleep ratio  
- Heart rate mean, standard deviation, and variability  
- Synthetic sleep-debt signal  
- Additional derived physiological metrics  

These features are normalized and prepared through a dedicated preprocessing pipeline.

---

## 7. Reinforcement Learning Formulation

### 7.1 PPO Agent
- Action space: 4 discrete intervention strategies  
- Policy: MLP with shared encoder  
- Advantage estimation via GAE  
- Optimized for behavioural decision boundaries  

### 7.2 SAC Agent
- Action space: continuous intervention intensities  
- Actor-critic architecture with entropy regularization  
- Suitable for fine-grained behaviour scaling  

### 7.3 Hybrid Model
- Learns a meta-policy that selects PPO or SAC  
- Combines discrete decision-making with continuous control  
- Demonstrates improved policy robustness  

---

## 8. Results Summary

Multiple experiments demonstrate that:

- PPO performs well for clear-cut decisions  
- SAC outperforms PPO when continuous adjustments are beneficial  
- The hybrid model yields the highest mean reward and stability across evaluation episodes  

These findings align with expected trade-offs between discrete and continuous RL methods.

---

## 9. License

This project is released under the MIT License.  
See the `LICENSE` file for details.

---

## 10. Author

Sri Sruthi Manikka Nagasamy  
Integrated MSc Data Science  
Reinforcement Learning and Predictive Modeling  
Personalized Health Intervention Research Project

---

## 11. Learning Outcomes

Through the development of this system, the following technical competencies were demonstrated:

- Ability to design and train deep learning models (LSTM) for multivariate time-series forecasting.
- Construction of a custom Gymnasium reinforcement learning environment modeled on real-world health metrics.
- Implementation and training of advanced RL algorithms including PPO (discrete control) and SAC (continuous control).
- Development of a hybrid policy-selection mechanism combining discrete and continuous RL for improved decision-making.
- Expertise in reward shaping, feature engineering, and trajectory evaluation for behavioural RL tasks.
- Integration of predictive modelling and reinforcement learning into a unified pipeline for personalized recommendations.
- Deployment of an interactive Streamlit interface for model demonstration and live recommendation generation.
- Experience structuring a research-grade machine learning codebase with modular components.
- Proficiency in data preprocessing techniques for wearable sensor datasets.
- Ability to compare algorithmic performance using standardized evaluation episodes, metrics, and visualization.
  
