import os
import numpy as np
import torch
from stable_baselines3 import PPO, DQN
import tensorflow as tf
from Health_Intervention_Agent_Production import (
    HealthDataProcessor,
    ProductionSleepPredictor,
    ProductionHybridAgent,
    ProductionRLTrainer,
)

# ------------------------------------------------------------
# 🔧 Environment setup
# ------------------------------------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✅ Using device: {device}")

# Models directory (absolute path safe)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# ------------------------------------------------------------
# 🧩 Initialize Data Processor
# ------------------------------------------------------------
processor = HealthDataProcessor()

# ------------------------------------------------------------
# 💤 Load pre-trained LSTM sleep predictor
# ------------------------------------------------------------
try:
    lstm_model_path = os.path.join(MODELS_DIR, "production_lstm_sleep_predictor.h5")
    lstm_predictor = tf.keras.models.load_model(lstm_model_path, compile=False)
    print(f"✅ Loaded LSTM model from {lstm_model_path}")
except Exception as e:
    print(f"⚠️ Could not load LSTM model: {e}")
    lstm_predictor = None

# ------------------------------------------------------------
# 🧠 Load pre-trained PPO and DQN agents
# ------------------------------------------------------------
ppo_model_path = os.path.join(MODELS_DIR, "ppo_health_agent.zip")
dqn_model_path = os.path.join(MODELS_DIR, "dqn_health_agent.zip")

ppo_model = None
dqn_model = None

try:
    if os.path.exists(ppo_model_path):
        ppo_model = PPO.load(ppo_model_path, device=device)
        print(f"✅ PPO model loaded from {ppo_model_path}")
    else:
        print("⚠️ PPO model not found — ensure it’s saved in models/")

    if os.path.exists(dqn_model_path):
        dqn_model = DQN.load(dqn_model_path, device=device)
        print(f"✅ DQN model loaded from {dqn_model_path}")
    else:
        print("⚠️ DQN model not found — ensure it’s saved in models/")

except Exception as e:
    print(f"❌ Error loading RL models: {e}")

# ------------------------------------------------------------
# 🔄 Create Hybrid Agent
# ------------------------------------------------------------
try:
    if ppo_model is not None and dqn_model is not None:
        hybrid_agent = ProductionHybridAgent(ppo_model, dqn_model, processor)
        print("✅ Hybrid agent initialized successfully!")
    else:
        hybrid_agent = None
        print("⚠️ Could not initialize hybrid agent (missing models)")
except Exception as e:
    hybrid_agent = None
    print(f"❌ Failed to initialize hybrid agent: {e}")

# ------------------------------------------------------------
# ✅ Export key objects for app.py
# ------------------------------------------------------------
__all__ = [
    "processor",
    "hybrid_agent",
    "lstm_predictor",
    "ppo_model",
    "dqn_model",
]
