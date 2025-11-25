#!/usr/bin/env python
# coding: utf-8
"""
Cleaned production-safe Health_Intervention_Agent_Production.py

- No heavy training runs at import
- Safe model-loading helper
- Core classes available for import
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ML / RL libs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import gymnasium as gym
from gymnasium import spaces

# stable-baselines3 imports wrapped in try/except — the environment using this module
# may not have these libs installed when importing only for the Streamlit UI.
try:
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except Exception:
    PPO = DQN = Monitor = DummyVecEnv = BaseCallback = None
    SB3_AVAILABLE = False

import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

# device auto-detection (MPS vs cpu)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)


# -------------------------
# Data Processor
# -------------------------
class HealthDataProcessor:
    """Production-level data processor with narrowing-RL helper functions."""
    def __init__(self):
        self.scaler = StandardScaler()
        self.user_profiles = {}
        self.similarity_matrix = None
        self.user_ids = []

    def load_and_preprocess_data(self,
                                 activity_path=None,
                                 sleep_path=None,
                                 heartrate_path=None):
        """
        Load Fitbit CSVs and perform light preprocessing.
        If paths are None, attempt to load from default hard-coded location.
        Returns (activity_df, daily_sleep_df, heartrate_df).
        """
        # You can override these paths when calling the function.
        # Defaults (user-specific) — update as needed.
        default_base = os.path.expanduser(
            "~/Downloads/Courses- Semester 7 (Soul Purpose)/22CSC403- Reinforcement Learning/Project/archive (5)/mturkfitbit_export_3.12.16-4.11.16/Fitabase Data 3.12.16-4.11.16"
        )

        if activity_path is None:
            activity_path = os.path.join(default_base, "dailyActivity_merged.csv")
        if sleep_path is None:
            sleep_path = os.path.join(default_base, "minuteSleep_merged.csv")
        if heartrate_path is None:
            heartrate_path = os.path.join(default_base, "heartrate_seconds_merged.csv")

        # Safe loading with try/except so import doesn't crash if files missing
        try:
            activity = pd.read_csv(activity_path)
            sleep = pd.read_csv(sleep_path)
            heartrate = pd.read_csv(heartrate_path)
        except Exception as e:
            # Return empty frames on failure and log warning
            print(f"⚠️ Could not load CSVs: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Preprocess activity
        if 'ActivityDate' in activity.columns:
            activity['ActivityDate'] = pd.to_datetime(activity['ActivityDate'])
            activity['day_of_week'] = activity['ActivityDate'].dt.dayofweek
            activity['is_weekend'] = activity['day_of_week'].isin([5, 6]).astype(int)
        else:
            # attempt common alternatives / fail gracefully
            activity['day_of_week'] = 0
            activity['is_weekend'] = 0

        # Aggregate minute-level sleep -> daily average state
        if 'date' in sleep.columns and 'value' in sleep.columns:
            sleep['date'] = pd.to_datetime(sleep['date'])
            daily_sleep = sleep.groupby(['Id', sleep['date'].dt.date]).agg({'value': 'mean'}).reset_index()
            daily_sleep.columns = ['Id', 'date', 'avg_sleep_state']
        else:
            daily_sleep = pd.DataFrame()

        return activity, daily_sleep, heartrate

    def create_user_profiles(self, activity_data):
        """Extract per-user summary features and compute a similarity matrix."""
        if activity_data is None or activity_data.empty:
            return {}

        user_features = []
        self.user_ids = []

        for user_id in activity_data['Id'].unique():
            user_data = activity_data[activity_data['Id'] == user_id]

            # compute robust stats with safe fallbacks
            try:
                avg_steps = user_data['TotalSteps'].mean()
            except Exception:
                avg_steps = 0.0
            try:
                avg_calories = user_data['Calories'].mean()
            except Exception:
                avg_calories = 0.0
            try:
                avg_active = (user_data.get('VeryActiveMinutes', 0).mean() +
                              user_data.get('FairlyActiveMinutes', 0).mean())
            except Exception:
                avg_active = 0.0
            try:
                avg_sedentary = user_data.get('SedentaryMinutes', 0).mean()
            except Exception:
                avg_sedentary = 0.0
            try:
                activity_consistency = user_data['TotalSteps'].std()
            except Exception:
                activity_consistency = 0.0
            try:
                weekend_mean = user_data[user_data.get('is_weekend', 0) == 1]['TotalSteps'].mean()
                weekday_mean = user_data[user_data.get('is_weekend', 0) == 0]['TotalSteps'].mean()
                weekend_activity_ratio = weekend_mean / (weekday_mean + 1e-6)
            except Exception:
                weekend_activity_ratio = 1.0

            profile = {
                'avg_steps': avg_steps,
                'avg_calories': avg_calories,
                'avg_active_minutes': avg_active,
                'avg_sedentary_minutes': avg_sedentary,
                'activity_consistency': activity_consistency,
                'weekend_activity_ratio': weekend_activity_ratio,
            }

            user_features.append(list(profile.values()))
            self.user_ids.append(user_id)
            self.user_profiles[user_id] = profile

        user_features = np.array(user_features)
        try:
            self.similarity_matrix = euclidean_distances(user_features)
        except Exception:
            # fallback: identity distances
            self.similarity_matrix = np.eye(len(self.user_ids))

        return self.user_profiles

    def get_similar_users(self, target_user_id, n_users=5):
        """Return N most similar users (excluding target)."""
        if not self.user_ids or self.similarity_matrix is None:
            return []
        try:
            idx = self.user_ids.index(target_user_id)
        except ValueError:
            return []
        sims = self.similarity_matrix[idx]
        similar_idx = np.argsort(sims)[1:n_users + 1]
        return [self.user_ids[i] for i in similar_idx]


# module-level processor instance
processor = HealthDataProcessor()


# -------------------------
# Production LSTM Sleep Predictor
# -------------------------
class ProductionSleepPredictor:
    """
    Production-grade LSTM model wrapper for sleep quality prediction.
    The model architecture is defined here; training is only invoked explicitly.
    """
    def __init__(self, sequence_length=7):
        self.sequence_length = sequence_length
        self.model = self._build_production_model()
        self.training_history = None
        self.last_predictions = None

    def _build_production_model(self):
        model = Sequential([
            Input(shape=(self.sequence_length, 4)),
            LSTM(128, return_sequences=True, dropout=0.2),
            LSTM(64, return_sequences=True, dropout=0.2),
            LSTM(32, return_sequences=False, dropout=0.2),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid'),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        return model

    def train(self, X_train, y_train, epochs=100, validation_split=0.2, callbacks=None):
        """Train model. Caller must provide data. Returns history."""
        if callbacks is None:
            callbacks = [
                EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss', min_delta=0.001),
                ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-6, monitor='val_loss')
            ]
        self.training_history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        # save model (caller can choose to skip)
        try:
            os.makedirs("models", exist_ok=True)
            self.model.save('models/production_lstm_sleep_predictor.h5')
        except Exception as e:
            print(f"⚠️ Could not save LSTM model: {e}")
        return self.training_history

    def predict(self, X):
        """Return model predictions (1D array)."""
        if self.model is None:
            raise RuntimeError("Model not built")
        preds = self.model.predict(X)
        return preds.flatten()


# -------------------------
# Gym Environment
# -------------------------
class HealthInterventionEnv(gym.Env):
    """Custom Gym environment representing user health state + interventions."""
    metadata = {"render.modes": ["human"]}

    def __init__(self, user_data, similar_users_data=None, lstm_predictor=None):
        super().__init__()
        # Action: 0=no-op, 1=walk, 2=meditate, 3=sleep early, 4=hydrate
        self.action_space = spaces.Discrete(5)
        # observation: steps, heart_rate, sleep_quality, day_of_week, fatigue, hydration
        self.observation_space = spaces.Box(
            low=np.array([0, 40, 0.3, 0, 0, 0], dtype=np.float32),
            high=np.array([30000, 180, 1.0, 6, 10, 100], dtype=np.float32),
            dtype=np.float32
        )
        self.user_data = user_data.reset_index(drop=True) if not user_data is None else pd.DataFrame()
        self.similar_users_data = similar_users_data or []
        self.lstm_predictor = lstm_predictor
        self.current_day = 0
        self.max_days = min(60, len(self.user_data)) if not self.user_data.empty else 30
        self.state = self._get_initial_state()
        self.episode_history = []
        self.last_action = None

    def _get_initial_state(self):
        if not self.user_data.empty and self.current_day < len(self.user_data):
            day_data = self.user_data.iloc[self.current_day]
            steps = day_data.get('TotalSteps', 6000)
            return np.array([steps, 70.0, 0.75, self.current_day % 7, 0.0, 70.0], dtype=np.float32)
        return np.array([6000., 70., 0.75, 0., 0., 70.], dtype=np.float32)

    def step(self, action):
        # Normalize action to scalar int
        if isinstance(action, (np.ndarray, list, tuple)):
            try:
                action = int(np.asarray(action).flatten()[0])
            except Exception:
                action = int(action[0]) if hasattr(action, '__len__') else int(action)
        action = int(action)
        action = max(0, min(4, action))

        current_state = self.state.copy()
        steps, heart_rate, sleep_quality, day_of_week, fatigue, hydration = current_state

        reward = self._calculate_reward(current_state, action)

        # Apply action effects (simple stochastic transitions)
        if action == 1:
            steps = min(steps + np.random.randint(1500, 3000), 25000)
            heart_rate = min(heart_rate + 8, 140)
            fatigue += 0.3
            hydration = max(hydration - 10, 0)
        elif action == 2:
            heart_rate = max(heart_rate - 12, 55)
            sleep_quality = min(sleep_quality + 0.08, 0.98)
            fatigue = max(fatigue - 0.2, 0)
        elif action == 3:
            sleep_quality = min(sleep_quality + 0.12, 0.98)
            fatigue = max(fatigue - 0.4, 0)
        elif action == 4:
            hydration = min(hydration + 25, 100)
            heart_rate = max(heart_rate - 5, 55)

        # Natural variation
        steps = max(steps + np.random.randint(-800, 1200), 1000)
        sleep_quality = max(min(sleep_quality + np.random.uniform(-0.05, 0.05), 1.0), 0.3)

        self.current_day += 1
        next_state = np.array([steps, heart_rate, sleep_quality, (day_of_week + 1) % 7, fatigue, hydration],
                              dtype=np.float32)
        self.state = next_state

        terminated = self.current_day >= self.max_days
        truncated = False

        self.episode_history.append({
            'day': self.current_day,
            'state': current_state,
            'action': action,
            'reward': reward,
            'next_state': next_state
        })

        self.last_action = action
        return next_state, reward, terminated, truncated, {}

    def _calculate_reward(self, state, action):
        steps, heart_rate, sleep_quality, day_of_week, fatigue, hydration = state
        reward = 0.0

        if 7000 <= steps <= 12000:
            reward += 2.0
        elif steps > 15000:
            reward -= 1.0
        elif steps < 3000:
            reward -= 0.5

        if sleep_quality > 0.85:
            reward += 2.5
        elif sleep_quality > 0.75:
            reward += 1.5
        elif sleep_quality < 0.5:
            reward -= 1.0

        if 60 <= heart_rate <= 75:
            reward += 1.5
        elif heart_rate > 100:
            reward -= 0.5

        if hydration > 70:
            reward += 0.8
        elif hydration < 40:
            reward -= 1.0

        action_penalties = {0: -0.1, 1: 0.0, 2: 0.0, 3: 0.0, 4: -0.2}
        reward += action_penalties.get(action, 0.0)

        # Contextual bonuses/penalties
        if action == 1:
            if steps < 5000:
                reward += 1.5
            elif steps > 12000:
                reward -= 0.8
            if fatigue > 5:
                reward -= 0.5
        elif action == 2:
            if heart_rate > 80:
                reward += 1.2
            if sleep_quality < 0.6:
                reward += 0.8
        elif action == 3:
            if sleep_quality < 0.7:
                reward += 1.5
            elif sleep_quality > 0.9:
                reward -= 0.5
            if day_of_week in [0, 1, 2, 3, 4]:
                reward += 0.3
            else:
                reward -= 0.2
        elif action == 4:
            if hydration < 60:
                reward += 1.0
            elif hydration > 85:
                reward -= 0.5

        reward -= fatigue * 0.1

        if hasattr(self, 'last_action') and action == self.last_action:
            reward -= 0.3

        return float(reward)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_day = 0
        self.state = self._get_initial_state()
        self.episode_history = []
        self.last_action = None
        return self.state, {}

    def get_episode_metrics(self):
        if not self.episode_history:
            return {}
        rewards = [s['reward'] for s in self.episode_history]
        states = [s['state'] for s in self.episode_history]
        return {
            'total_reward': float(sum(rewards)),
            'average_reward': float(np.mean(rewards)) if rewards else 0.0,
            'steps_taken': float(np.mean([s[0] for s in states])) if states else 0.0,
            'sleep_quality': float(np.mean([s[2] for s in states])) if states else 0.0,
            'episode_length': len(self.episode_history)
        }


# -------------------------
# Production RL Trainer (safe)
# -------------------------
class ProductionRLTrainer:
    """
    RL trainer wrapper. This class does not start training automatically.
    Use its methods explicitly when training is desired.
    """
    def __init__(self, data_processor, lstm_predictor=None):
        self.processor = data_processor
        self.lstm_predictor = lstm_predictor
        self.training_metrics = []

    def create_environment(self, user_id, use_narrowing=True):
        activity, _, _ = self.processor.load_and_preprocess_data()
        if activity.empty:
            raise RuntimeError("Activity data not available to create environment.")
        user_data = activity[activity['Id'] == user_id]
        similar_users_data = []
        if use_narrowing:
            similar_users = self.processor.get_similar_users(user_id, n_users=3)
            for su in similar_users:
                similar_users_data.append(activity[activity['Id'] == su])
        env = HealthInterventionEnv(user_data, similar_users_data, self.lstm_predictor)
        return env

    def train_ppo_production(self, user_id, total_timesteps=100000, **ppo_kwargs):
        if not SB3_AVAILABLE:
            raise RuntimeError("stable-baselines3 (PPO) not available in this environment.")
        env = self.create_environment(user_id, use_narrowing=True)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        model = PPO("MlpPolicy", env, device=device, verbose=1, **ppo_kwargs)
        # user must call model.learn(...) explicitly (we don't call learn here)
        return model, env

    def train_dqn_production(self, user_id, total_timesteps=100000, **dqn_kwargs):
        if not SB3_AVAILABLE:
            raise RuntimeError("stable-baselines3 (DQN) not available in this environment.")
        env = self.create_environment(user_id, use_narrowing=True)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        model = DQN("MlpPolicy", env, device=device, verbose=1, **dqn_kwargs)
        return model, env


# -------------------------
# Hybrid Agent
# -------------------------
class ProductionHybridAgent:
    """
    Simple hybrid wrapper that picks between PPO and DQN (or falls back).
    This is a thin orchestration layer; model updating / fine-tuning must be done externally.
    """
    def __init__(self, ppo_model=None, dqn_model=None, data_processor=None):
        self.ppo_model = ppo_model
        self.dqn_model = dqn_model
        self.processor = data_processor or processor
        self.user_feedback = []
        self.personalization_level = 0.0
        self.performance_history = []

    def _choose_model_for_inference(self, deterministic=True):
        # Prioritize PPO if available (stable choice), else DQN, else None
        if self.ppo_model is not None:
            return self.ppo_model
        if self.dqn_model is not None:
            return self.dqn_model
        return None

    def predict(self, observation, user_id=None, deterministic=True):
        # Accept lists/ndarrays; convert to single observation if needed
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        model = self._choose_model_for_inference(deterministic)
        if model is None:
            # fallback random action
            return int(np.random.randint(0, 5))

        # stable-baselines3 .predict expects vectorized obs in some cases;
        # we call with deterministic arg when available
        try:
            # many sb3 policies return (action, state) tuple
            out = model.predict(obs, deterministic=deterministic)
            if isinstance(out, tuple):
                action = out[0]
            else:
                action = out
            # action may be array-like
            if isinstance(action, np.ndarray):
                action = int(action.flatten()[0])
            elif isinstance(action, (list, tuple)):
                action = int(action[0])
            else:
                action = int(action)
            return max(0, min(4, action))
        except Exception:
            return int(np.random.randint(0, 5))

    def add_feedback(self, user_id, action, state, reward, feedback_score):
        entry = {
            'user_id': user_id,
            'action': int(action),
            'state': np.array(state).tolist() if isinstance(state, np.ndarray) else state,
            'reward': float(reward),
            'feedback_score': float(feedback_score),
            'timestamp': pd.Timestamp.now(),
            'personalization_level': self.personalization_level
        }
        self.user_feedback.append(entry)
        self.personalization_level = min(1.0, len(self.user_feedback) / 15.0)
        self.performance_history.append({
            'timestamp': pd.Timestamp.now(),
            'personalization_level': self.personalization_level,
            'feedback_score': feedback_score,
            'total_feedback': len(self.user_feedback)
        })

    def plot_learning_progress(self):
        """Visualize learning and personalization progress over time."""
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np

        if not hasattr(self, 'performance_history') or not self.performance_history:
            print("No performance history available.")
            return None

        df = pd.DataFrame(self.performance_history)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1️⃣ Personalization Level Progress
        axes[0, 0].plot(df['timestamp'], df['personalization_level'], 'o-', linewidth=3)
        axes[0, 0].set_title('Personalization Progress')
        axes[0, 0].set_ylabel('Personalization Level')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0.5, color='red', linestyle='--', label='Cold Start Threshold')
        axes[0, 0].legend()

        # 2️⃣ Feedback Accumulation
        axes[0, 1].plot(df['timestamp'], df['total_feedback'], 's-', linewidth=2, color='green')
        axes[0, 1].set_title('Feedback Accumulation')
        axes[0, 1].set_ylabel('Total Feedback')
        axes[0, 1].grid(True, alpha=0.3)

        # 3️⃣ Feedback Quality Trend
        axes[1, 0].plot(df['timestamp'], df['feedback_score'], '^-', linewidth=2, color='orange')
        axes[1, 0].set_title('Feedback Quality Over Time')
        axes[1, 0].set_ylabel('Feedback Score')
        axes[1, 0].grid(True, alpha=0.3)

        # 4️⃣ Personalization vs Feedback Volume
        axes[1, 1].scatter(df['total_feedback'], df['personalization_level'], s=80, alpha=0.7)
        axes[1, 1].set_title('Personalization vs Feedback Volume')
        axes[1, 1].set_xlabel('Total Feedback')
        axes[1, 1].set_ylabel('Personalization Level')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def get_metrics(self):
        avg_feedback = np.mean([f['feedback_score'] for f in self.user_feedback]) if self.user_feedback else 0.0
        return {
            'total_feedback': len(self.user_feedback),
            'personalization_level': self.personalization_level,
            'average_feedback_score': float(avg_feedback),
            'algorithm': 'Hybrid (PPO + DQN + Narrowing RL)',
            'cold_start_overcome': self.personalization_level > 0.5
        }


# -------------------------
# Utility: Safe model loader
# -------------------------
def load_models(models_dir="models"):
    """
    Attempt to load LSTM (Keras) and RL agents (stable-baselines3) from `models_dir`.
    Returns (lstm_model, ppo_model, dqn_model). Missing models are returned as None.
    """
    lstm_model = None
    ppo_model = None
    dqn_model = None

    # LSTM
    try:
        lstm_path = os.path.join(models_dir, "production_lstm_sleep_predictor.h5")
        if os.path.isfile(lstm_path):
            # load with compile=False to avoid custom-object deserialization issues
            lstm_model = tf.keras.models.load_model(lstm_path, compile=False)
    except Exception as e:
        print(f"⚠️ Could not load LSTM model: {e}")
        lstm_model = None

    # PPO / DQN (stable-baselines3 ZIP or folder)
    if SB3_AVAILABLE:
        try:
            # attempt common filenames
            ppo_candidates = [
                os.path.join(models_dir, "ppo_health_agent.zip"),
                os.path.join(models_dir, "ppo_health_agent_production_1503960366.zip"),
            ]
            for c in ppo_candidates:
                if os.path.isfile(c):
                    ppo_model = PPO.load(c, device=device)
                    break
        except Exception as e:
            print(f"⚠️ Could not load PPO model: {e}")
            ppo_model = None

        try:
            dqn_candidates = [
                os.path.join(models_dir, "dqn_health_agent.zip"),
                os.path.join(models_dir, "dqn_health_agent_production_1503960366.zip"),
            ]
            for c in dqn_candidates:
                if os.path.isfile(c):
                    dqn_model = DQN.load(c, device=device)
                    break
        except Exception as e:
            print(f"⚠️ Could not load DQN model: {e}")
            dqn_model = None

    return lstm_model, ppo_model, dqn_model


# Expose a safe set of top-level variables (no training executed)
production_lstm = None
ppo_model = None
dqn_model = None
trainer = None
hybrid_agent = None

# Try to load existing models on demand but do not crash on import
try:
    production_lstm, ppo_model, dqn_model = load_models(models_dir="models")
except Exception:
    production_lstm = ppo_model = dqn_model = None

# create trainer/hybrid agent if models or processor exist
try:
    trainer = ProductionRLTrainer(processor, lstm_predictor=None)
except Exception:
    trainer = None

try:
    hybrid_agent = ProductionHybridAgent(ppo_model=ppo_model, dqn_model=dqn_model, data_processor=processor)
except Exception:
    hybrid_agent = None


# -------------------------
# Safe CLI/demo: runs only when executed directly
# -------------------------
if __name__ == "__main__":
    print("✅ Module executed directly. Demo mode (no training will run).")
    print(f"Device: {device}")
    print("Processor ready:", isinstance(processor, HealthDataProcessor))
    print("Loaded models:", {
        'lstm': bool(production_lstm),
        'ppo': bool(ppo_model),
        'dqn': bool(dqn_model),
    })

    # Load data summary (safe)
    activity, daily_sleep, hr = processor.load_and_preprocess_data()
    print(f"Activity rows: {len(activity)}, Sleep rows: {len(daily_sleep)}, HR rows: {len(hr)}")

    # Example: build user profiles if data exists
    if not activity.empty:
        processor.create_user_profiles(activity)
        uids = activity['Id'].unique()[:5].tolist()
        print("Example users:", uids)
    else:
        print("No activity data found — place CSVs in expected location or call load_and_preprocess_data with paths.")

    print("Module ready for import into Streamlit / health_agent_core.")
