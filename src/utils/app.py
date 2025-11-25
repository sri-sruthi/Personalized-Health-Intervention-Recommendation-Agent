import streamlit as st
import traceback
import numpy as np
import pandas as pd
from datetime import datetime

from health_agent_core import (
    processor,
    hybrid_agent as core_agent,
    ProductionSleepPredictor,
)

# ------------------------------------------------------------
# Streamlit Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Personalized Health Intervention Agent",
    layout="wide",
)

# ------------------------------------------------------------
# Maintain Agent State Between Reruns
# ------------------------------------------------------------
if "hybrid_agent" not in st.session_state:
    st.session_state["hybrid_agent"] = core_agent
hybrid_agent = st.session_state["hybrid_agent"]

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
def simulate_current_state():
    """Generate a simulated daily health state for demonstration."""
    return np.array([
        np.random.randint(4000, 12000),
        np.random.normal(70, 8),
        np.random.uniform(0.6, 0.9),
        np.random.randint(0, 7),
        np.random.uniform(0, 3),
        np.random.uniform(50, 90)
    ])

def interpret_action(action):
    """Convert numeric action to a readable recommendation."""
    descriptions = {
        0: "No action needed — your metrics look balanced.",
        1: "Take a 20-minute walk — great for circulation and focus.",
        2: "Meditate for 10 minutes — reduce stress and boost calm.",
        3: "Sleep early tonight — recovery matters.",
        4: "Drink 2 glasses of water — stay hydrated."
    }
    return descriptions.get(action, "No recommendation available.")

def record_feedback(agent, user_id, action, state, feedback_score):
    """Record feedback and update agent personalization."""
    reward = feedback_score * 2.0
    agent.add_feedback(user_id, action, state, reward, feedback_score)
    print(f"Feedback recorded for {user_id}: {feedback_score}")
    print(f"Total feedback so far: {len(agent.user_feedback)}")
    st.session_state["hybrid_agent"] = agent  # persist
    st.success("Feedback recorded successfully!")

# ------------------------------------------------------------
# UI Layout
# ------------------------------------------------------------
st.title("Personalized Health Intervention Agent")
st.markdown(
    "This system combines Deep Learning and Reinforcement Learning "
    "to guide healthier daily routines using wearable data."
)

# Sidebar
st.sidebar.header("User Settings")
user_list = ["User A", "User B", "User C"]
selected_user = st.sidebar.selectbox("Select a user profile", user_list)

st.sidebar.divider()
st.sidebar.markdown("Dashboard Options")
dashboard_mode = st.sidebar.radio(
    "View mode:",
    ["Get AI Recommendation", "View Agent Metrics", "About the System"]
)

# ------------------------------------------------------------
# 1. Recommendation Mode
# ------------------------------------------------------------
if dashboard_mode == "Get AI Recommendation":
    st.subheader("Health Recommendation")

    if st.button("Generate Recommendation"):
        try:
            state = simulate_current_state()
            action = hybrid_agent.predict(state, user_id=selected_user, deterministic=True)
            recommendation = interpret_action(action)

            st.session_state["last_state"] = state
            st.session_state["last_action"] = action
            st.session_state["last_user"] = selected_user

            st.markdown("Current State:")
            st.write(pd.DataFrame([state], columns=[
                "Steps", "Heart Rate", "Sleep Quality", "Day of Week", "Fatigue", "Hydration"
            ]))

            st.markdown(f"Recommendation: **{recommendation}**")

        except Exception as e:
            st.error("An error occurred while generating recommendation.")
            st.code(traceback.format_exc())

    # Feedback section
    if "last_state" in st.session_state:
        st.markdown("Was this recommendation helpful?")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Helpful"):
                record_feedback(hybrid_agent,
                                st.session_state["last_user"],
                                st.session_state["last_action"],
                                st.session_state["last_state"],
                                1.0)
                st.rerun()
        with col2:
            if st.button("Neutral"):
                record_feedback(hybrid_agent,
                                st.session_state["last_user"],
                                st.session_state["last_action"],
                                st.session_state["last_state"],
                                0.5)
                st.rerun()
        with col3:
            if st.button("Not Helpful"):
                record_feedback(hybrid_agent,
                                st.session_state["last_user"],
                                st.session_state["last_action"],
                                st.session_state["last_state"],
                                0.0)
                st.rerun()

# ------------------------------------------------------------
# 2. Metrics Mode
# ------------------------------------------------------------
elif dashboard_mode == "View Agent Metrics":
    st.subheader("Agent Performance Metrics")

    try:
        metrics = hybrid_agent.get_metrics()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Feedback", metrics["total_feedback"])
        col2.metric("Average Feedback Score", f"{metrics['average_feedback_score']:.2f}")
        col3.metric("Personalization Level", f"{metrics['personalization_level']*100:.1f}%")

        st.markdown(f"Algorithm: {metrics['algorithm']}")
        st.markdown(f"Cold Start Overcome: {'Yes' if metrics['cold_start_overcome'] else 'No'}")

        import matplotlib.pyplot as plt

        if metrics["total_feedback"] > 0:
            st.markdown("#### 📊 Personalization Progress:")
        fig = hybrid_agent.plot_learning_progress()
        if fig:
            st.pyplot(fig)
            plt.close(fig)


    except Exception as e:
        st.error("Failed to load metrics.")
        st.code(traceback.format_exc())

# ------------------------------------------------------------
# 3. About Mode
# ------------------------------------------------------------
else:
    st.subheader("About the System")
    st.markdown("""
    This personalized health intervention system combines:
    - LSTM-based sleep prediction for forecasting sleep efficiency  
    - PPO + DQN hybrid reinforcement learning for adaptive health guidance  
    - Real-time feedback for personalization  
    - Streamlit dashboard for visualization  

    Key Features:
    - Hybrid PPO + DQN learning architecture  
    - Cold-start resolution via Narrowing RL  
    - Continuous personalization through feedback  
    - Optimized for Apple Silicon (MPS)
    """)

st.sidebar.divider()
st.sidebar.caption("Health Intervention Agent — Reinforcement Learning Powered")
