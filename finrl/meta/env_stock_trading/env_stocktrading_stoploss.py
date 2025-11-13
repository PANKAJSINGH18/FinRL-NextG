# train_agent_parallel.py
import os
import sys
import wandb
import numpy as np
import pandas as pd

# Define constants at the top level
COIN_NAME = "etherium" # Make sure this is correct
DATA_OUTPUT_PATH = "research/data"
CHART_PATH = f"research/charts/{COIN_NAME}"
DATA_PATH = None
SANITY_CHECK = True

RUN_LOCAL_TEST = not os.path.exists("/kaggle/input")

if not RUN_LOCAL_TEST:
    import subprocess
    from kaggle_secrets import UserSecretsClient

    SANITY_CHECK = False
    # 4️⃣  Install FinRL package in editable mode (so imports work)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--upgrade", "pip"
    ])

    # subprocess.check_call([
    #     sys.executable, "-m", "pip", "install", "git+https://github.com/ipankaj18/FinRL-NextG.git", "--quiet"
    # ])

    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "git+https://github.com/ipankaj18/FinRL-NextG.git@ipankaj18-patch-8",
        "--quiet"
    ])

    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "protobuf==3.20.3", "--force-reinstall"
    ])

    subprocess.check_call([
        sys.executable,
        "-m", "pip", "install",
        "stable-baselines3[extra] @ git+https://github.com/DLR-RM/stable-baselines3"
    ])

    secrets = UserSecretsClient()
    wandb_api_key = secrets.get_secret("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)
    DATA_PATH = f"/kaggle/input/crypto-data-5y/{COIN_NAME}.csv"
    os.makedirs(DATA_OUTPUT_PATH, exist_ok=True)
    os.makedirs(CHART_PATH, exist_ok=True)
else:
    try:
        from config.config import DATA_PATH as BASE_DATA_PATH, CHART_PATH as BASE_CHART
        DATA_PATH = f"{BASE_DATA_PATH}/{COIN_NAME}.csv"
        CHART_PATH = os.path.join(BASE_CHART, COIN_NAME)
    except Exception as e:
        print(f"Error loading config: {e}. Using fallback paths.")
        DATA_PATH = f"data/{COIN_NAME}.csv"
        CHART_PATH = f"research/charts/{COIN_NAME}"
    os.makedirs(CHART_PATH, exist_ok=True)

import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv # Use SubprocVecEnv
from finrl.meta.env_stock_trading.env_stocktrading_stoploss import StockTradingEnvStopLoss
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
# =========================================
# 2️⃣  Load Your CSV Data (Outside main guard)
# =========================================
df = pd.read_csv(DATA_PATH)
if SANITY_CHECK:
    df = df.head(1000)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.rename(columns={"timestamp": "date"})
df = df.sort_values(["date"]).reset_index(drop=True)
df["tic"] = COIN_NAME

required_cols = ["date", "tic", "open", "high", "low", "close", "volume"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

indicator_list = ["macd", "rsi", "cci", "adx"]

fe = FeatureEngineer(use_technical_indicator=True, tech_indicator_list=indicator_list)
df = fe.preprocess_data(df)

train_df = df[(df["date"] >= pd.Timestamp("2020-03-03")) & (df["date"] <= pd.Timestamp("2023-12-31"))].reset_index(drop=True)
test_df  = df[(df["date"] >= pd.Timestamp("2024-01-01")) & (df["date"] <= pd.Timestamp("2025-03-02"))].reset_index(drop=True)

# Compute state space correctly
stock_dim = len(train_df["tic"].unique())
tech_dim = len(indicator_list)
state_space = 1 + 2 * stock_dim + tech_dim  # [balance] + [price, shares] + [indicators]

# =========================================
# 3️⃣  Environment Factory Function (Outside main guard)
# =========================================
env_kwargs = {
    "initial_amount": 100000,
    # Add other kwargs if needed
}

def make_env(rank):
    """
    Factory function for creating environments for SubprocVecEnv.
    Each process gets its own instance of the environment.
    This function must be defined *before* the main execution block
    and be importable by child processes.
    """
    def _init():
        env = StockTradingEnvStopLoss(df=train_df, **env_kwargs)
        return Monitor(env) # Wrap individual env before vectorizing
    return _init

# =========================================
# 4️⃣  Custom W&B Callback (Outside main guard)
# =========================================
class WandbCallback(BaseCallback):
    """
    Logs detailed training metrics and debug info from Stable-Baselines3 into Weights & Biases.
    """

    def __init__(self, log_freq=1000, verbose=1):
        super(WandbCallback, self).__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.mean_rewards = []
        self.episode_count = 0

    def _on_training_start(self):
        if self.verbose:
            print("🚀 Training started — W&B logging active.")
        wandb.log({"training_started": True})

    def _on_step(self) -> bool:
        """
        Called at every environment step.
        Logs metrics every `log_freq` steps.
        """
        # Extract reward safely - SB3 VecEnv returns vectorized rewards
        reward = self.locals.get("rewards")
        if reward is not None:
            if isinstance(reward, (list, np.ndarray)):
                mean_reward = np.mean(reward) # Average across all envs
            else:
                mean_reward = reward
            wandb.log({"train/reward_step": mean_reward})

        # Track per-step debug info
        if self.n_calls % self.log_freq == 0:
            metrics = {
                "train/steps": self.num_timesteps,
                "train/mean_reward": np.mean(self.mean_rewards) if self.mean_rewards else 0
            }

            # Try to fetch more internal details (depends on algorithm)
            try:
                if hasattr(self.model, "logger"):
                    log_dict = dict(self.model.logger.name_to_value)
                    for k, v in log_dict.items():
                        if isinstance(v, (int, float, np.number)):
                            metrics[f"model/{k}"] = v
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Could not log model internals: {e}")

            wandb.log(metrics)

            if self.verbose:
                print(f"[Step {self.num_timesteps}] Mean reward: {metrics['train/mean_reward']:.4f}")

        return True

    def _on_rollout_end(self):
        """
        Called when a rollout ends — track episode info.
        """
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info.keys():
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                self.episode_count += 1
                wandb.log({
                    "episode/reward": ep_reward,
                    "episode/length": ep_length,
                    "episode/count": self.episode_count
                })
                if self.verbose:
                    print(f"🎯 Episode {self.episode_count}: reward={ep_reward:.2f}, length={ep_length}")

    def on_training_end(self) -> None:
        """
        Called when training is finished.
        """
        wandb.log({
            "training_done": True,
            "final/mean_reward": np.mean(self.mean_rewards) if self.mean_rewards else 0
        })
        if self.verbose:
            print("🏁 Training finished — all metrics logged to W&B.")


# =========================================
# 5️⃣  Main Execution Block (REQUIRED for multiprocessing)
# =========================================
if __name__ == '__main__': # <--- CRITICAL: Wrap all execution logic here
    # --- Optimization: Use multiple environments with SubprocVecEnv ---
    num_cpu = 8 # Number of processes to use. Adjust based on your CPU cores and memory.
    print(f"Creating {num_cpu} environments for training (using SubprocVecEnv)...")

    # Create a list of environment factory functions
    env_fns = [make_env(i) for i in range(num_cpu)]

    # Use SubprocVecEnv for parallel environments
    # start_method="fork" is usually faster on Linux/macOS
    # Use start_method="spawn" if you encounter issues on Windows or specific setups
    # Create parallel environments
    env_fns = [make_env(i) for i in range(num_cpu)]
    train_env = SubprocVecEnv(env_fns, start_method="fork")
    train_env = VecMonitor(train_env)

    # Keep the test environment single for evaluation
    test_env  = Monitor(StockTradingEnvStopLoss(df=test_df, **env_kwargs))

    # =========================================
    # 1️⃣  Initialize W&B
    # =========================================
    wandb.init(
        project="finrl-multi-env",      # your W&B project name
        config={
            "algorithm": "PPO",
            "total_timesteps": 200_000,  # Reduced for testing, increase as needed
            "num_parallel_envs": num_cpu,
            "n_steps_per_env": 512,      # Adjusted for 8 envs (512 * 8 = 4096 total)
            "batch_size": 1024,          # Increased batch size
            "optimized_environment": True,
            **env_kwargs
        }
    )

    # =========================================
    # 5️⃣  Create & Train Agent (Optimized for GPU with multiple envs)
    # =========================================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

    # *** CRITICAL ADJUSTMENT: n_steps ***
    # Total samples per update = n_steps * num_cpu
    # Original script: n_steps=4096 for 1 env => 4096 total samples per update
    # New script: n_steps=1024 for 4 envs => 1024 * 4 = 4096 total samples per update
    # Optimized PPO configuration
    model = PPO(
        "MlpPolicy",
        train_env,
        device=device,
        verbose=1,
        tensorboard_log=None,
        n_steps=512,        # Steps per environment (512 * 8 = 4096 total)
        batch_size=1024,    # Larger batch size for better GPU utilization
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=3e-4,
        policy_kwargs={
            'net_arch': [256, 256],  # Larger network for more GPU work
            'activation_fn': torch.nn.ReLU,
        }
    )

    # --- Optimization: Train the model ---
    trained_model = model.learn(
        total_timesteps=wandb.config["total_timesteps"],
        callback=[WandbCallback()],
        progress_bar=True
    )

    # =========================================
    # 6️⃣  Evaluate / Test (Uses single environment)
    # =========================================
    obs, info = test_env.reset()
    done = False
    episode_reward = 0
    steps = 0
    
    while not done and steps < 10000:  # Safety limit
        action, _ = trained_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        episode_reward += reward
        steps += 1
    
    final_value = getattr(test_env, "end_total_asset", 
                         test_env.account_information["total_assets"][-1] 
                         if hasattr(test_env, 'account_information') and test_env.account_information["total_assets"] 
                         else 0)
    
    wandb.log({
        "test/final_portfolio_value": final_value,
        "test/episode_reward": episode_reward,
        "test/steps": steps
    })
    
    # =========================================
    # SAVE MODEL
    # =========================================
    model_path = "optimized_trading_agent.zip"
    trained_model.save(model_path)
    wandb.save(model_path)
    
    wandb.finish()
    print("✅ OPTIMIZED training complete! Model saved & logged to W&B.")
