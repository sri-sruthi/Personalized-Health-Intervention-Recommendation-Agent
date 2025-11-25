import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import os

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.user_profiles = {}
        
    def load_data(self):
        """Load and merge all Fitbit datasets"""
        # Use your exact file paths
        activity = pd.read_csv('/Users/srisruthi/Downloads/Courses- Semester 7 (Soul Purpose)/22CSC403- Reinforcement Learning/Project/archive (5)/mturkfitbit_export_3.12.16-4.11.16/Fitabase Data 3.12.16-4.11.16/dailyActivity_merged.csv')
        sleep = pd.read_csv('/Users/srisruthi/Downloads/Courses- Semester 7 (Soul Purpose)/22CSC403- Reinforcement Learning/Project/archive (5)/mturkfitbit_export_3.12.16-4.11.16/Fitabase Data 3.12.16-4.11.16/minuteSleep_merged.csv')
        heartrate = pd.read_csv('/Users/srisruthi/Downloads/Courses- Semester 7 (Soul Purpose)/22CSC403- Reinforcement Learning/Project/archive (5)/mturkfitbit_export_3.12.16-4.11.16/Fitabase Data 3.12.16-4.11.16/heartrate_seconds_merged.csv')
        
        return activity, sleep, heartrate
    
    def create_user_profiles(self, activity_data):
        """Create user profiles for similarity comparison"""
        user_features = []
        user_ids = []
        
        for user_id in activity_data['Id'].unique():
            user_data = activity_data[activity_data['Id'] == user_id]
            
            # Extract key features for similarity
            profile = {
                'avg_steps': user_data['TotalSteps'].mean(),
                'avg_calories': user_data['Calories'].mean(),
                'active_minutes': user_data['VeryActiveMinutes'].mean() + user_data['FairlyActiveMinutes'].mean(),
                'sedentary_minutes': user_data['SedentaryMinutes'].mean()
            }
            
            user_features.append(list(profile.values()))
            user_ids.append(user_id)
            self.user_profiles[user_id] = profile
            
        # Calculate similarity matrix
        user_features = np.array(user_features)
        self.similarity_matrix = euclidean_distances(user_features)
        self.user_ids = user_ids
        
        return self.user_profiles
    
    def get_similar_users(self, target_user_id, n_users=5):
        """Find most similar users using Euclidean distance"""
        if target_user_id not in self.user_ids:
            return []
            
        target_idx = self.user_ids.index(target_user_id)
        similarities = self.similarity_matrix[target_idx]
        
        # Get indices of most similar users (excluding self)
        similar_indices = np.argsort(similarities)[1:n_users+1]
        similar_users = [self.user_ids[i] for i in similar_indices]
        
        return similar_users
    
    def prepare_lstm_data(self, sequence_length=7):
        """Prepare sequential data for LSTM sleep prediction"""
        activity, sleep, heartrate = self.load_data()
        
        # Simple feature engineering for demo
        activity['date'] = pd.to_datetime(activity['ActivityDate'])
        daily_features = activity.groupby(['Id', 'date']).agg({
            'TotalSteps': 'sum',
            'Calories': 'sum',
            'VeryActiveMinutes': 'sum',
            'SedentaryMinutes': 'sum'
        }).reset_index()
        
        return daily_features