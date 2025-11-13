from __future__ import annotations

import random
import time
import sys
from copy import deepcopy

import gymnasium as gym
import matplotlib
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
import logging
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger(__name__)


class StockTradingEnvStopLoss(gym.Env):
    """
    HEAVILY OPTIMIZED VERSION:
    - Zero pandas operations during step()
    - All data precomputed to numpy arrays
    - Vectorized operations
    - Memory reuse
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        buy_cost_pct=3e-3,
        sell_cost_pct=3e-3,
        date_col_name="date",
        hmax=10,
        discrete_actions=False,
        shares_increment=1,
        stoploss_penalty=0.9,
        profit_loss_ratio=2,
        turbulence_threshold=None,
        print_verbosity=100,  # Reduced frequency
        initial_amount=1e6,
        daily_information_cols=["open", "close", "high", "low", "volume"],
        cache_indicator_data=True,
        cash_penalty_proportion=0.1,
        random_start=True,
        patient=False,
        currency="$",
    ):
        # Store parameters
        self.df = df
        self.stock_col = "tic"
        self.random_start = random_start
        self.discrete_actions = discrete_actions
        self.patient = patient
        self.currency = currency
        self.shares_increment = shares_increment
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.print_verbosity = print_verbosity
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.stoploss_penalty = stoploss_penalty
        self.min_profit_penalty = 1 + profit_loss_ratio * (1 - self.stoploss_penalty)
        self.turbulence_threshold = turbulence_threshold
        self.daily_information_cols = daily_information_cols
        self.cash_penalty_proportion = cash_penalty_proportion
        
        # ===== CRITICAL OPTIMIZATION: Convert DataFrame to numpy =====
        self._precompute_all_data(df, date_col_name)
        
        # State and action spaces
        self.state_space = 1 + self.num_assets + self.num_assets * len(daily_information_cols)
        print(f"   - State size: {self.state_space}")
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_assets,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        
        self.turbulence = 0
        self.episode = -1
        self.episode_history = []
        self.printed_header = False
        
        # ===== OPTIMIZATION: Reusable arrays to avoid allocations =====
        self._reusable_arrays = {
            'zeros_assets': np.zeros(self.num_assets, dtype=np.float32),
            'ones_assets': np.ones(self.num_assets, dtype=np.float32),
        }

    def _precompute_all_data(self, df, date_col_name):
        """Convert pandas DataFrame to numpy arrays for maximum performance"""
        print("🔄 Precomputing environment data to numpy...")
        start_time = time.time()
        
        # Get unique dates and assets
        self.dates = df[date_col_name].sort_values().unique()
        self.assets = df[self.stock_col].unique()
        self.num_dates = len(self.dates)
        self.num_assets = len(self.assets)
        
        # Create fast lookup structures
        self.date_to_index = {date: idx for idx, date in enumerate(self.dates)}
        self.asset_to_index = {asset: idx for idx, asset in enumerate(self.assets)}
        
        # Pre-allocate data tensor: [dates, assets, features]
        self.data_tensor = np.zeros(
            (self.num_dates, self.num_assets, len(self.daily_information_cols)), 
            dtype=np.float32
        )
        
        # Fill data tensor from DataFrame
        df_indexed = df.set_index(date_col_name)
        for i, date in enumerate(self.dates):
            date_str = str(date)
            if date_str in df_indexed.index:
                day_data = df_indexed.loc[[date_str]]
                for j, asset in enumerate(self.assets):
                    asset_data = day_data[day_data[self.stock_col] == asset]
                    if len(asset_data) > 0:
                        for k, col in enumerate(self.daily_information_cols):
                            self.data_tensor[i, j, k] = asset_data[col].iloc[0]
        
        # Precompute close prices for fast access
        close_idx = self.daily_information_cols.index('close')
        self.close_prices = self.data_tensor[:, :, close_idx].copy()
        
        # Precompute all state vectors
        self.state_vectors = np.zeros(
            (self.num_dates, self.num_assets * len(self.daily_information_cols)), 
            dtype=np.float32
        )
        for i in range(self.num_dates):
            self.state_vectors[i] = self.data_tensor[i].flatten()
        
        print(f"✅ Data precomputed in {time.time() - start_time:.2f}s")
        print(f"   - Dates: {self.num_dates}, Assets: {self.num_assets}")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def current_step(self):
        return self.date_index - self.starting_point

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
            
        # Reset tracking variables with reusable arrays
        self.sum_trades = 0
        self.actual_num_trades = 0
        self.closing_diff_avg_buy = self._reusable_arrays['zeros_assets'].copy()
        self.profit_sell_diff_avg_buy = self._reusable_arrays['zeros_assets'].copy()
        self.n_buys = self._reusable_arrays['zeros_assets'].copy()
        self.avg_buy_price = self._reusable_arrays['zeros_assets'].copy()
        
        # Set starting point
        if self.random_start:
            self.starting_point = random.choice(range(int(self.num_dates * 0.5)))
        else:
            self.starting_point = 0
            
        self.date_index = self.starting_point
        self.turbulence = 0
        self.episode += 1
        
        # Initialize memory
        self.actions_memory = []
        self.transaction_memory = []
        self.state_memory = []
        self.account_information = {
            "cash": [],
            "asset_value": [],
            "total_assets": [],
            "reward": [],
        }
        
        # Create initial state
        state = np.zeros(self.state_space, dtype=np.float32)
        state[0] = self.initial_amount  # cash
        # holdings are zeros (positions 1:1+num_assets)
        state[1 + self.num_assets:] = self.state_vectors[self.date_index]  # market data
        
        self.state_memory.append(state)
        
        return state, {}

    def get_date_vector(self, date_index, cols=None):
        """Optimized: Direct array access"""
        return self.state_vectors[date_index].copy()

    def get_close_prices(self, date_index):
        """Optimized: Direct close prices access"""
        return self.close_prices[date_index].copy()

    def return_terminal(self, reason="Last Date", reward=0):
        state = self.state_memory[-1]
        self.log_step(reason=reason, terminal_reward=reward)
        
        # Log metrics
        total_assets = self.account_information["total_assets"][-1]
        gl_pct = total_assets / self.initial_amount
        
        logger.info("environment/GainLoss_pct: %.2f", (gl_pct - 1) * 100)
        logger.info("environment/total_assets: %d", int(total_assets))
        logger.info("environment/total_trades: %d", self.sum_trades)
        logger.info("environment/actual_num_trades: %d", self.actual_num_trades)
        
        if self.current_step > 0:
            logger.info("environment/avg_daily_trades: %.2f", self.sum_trades / self.current_step)
        
        logger.info("environment/completed_steps: %d", self.current_step)
        
        terminated = True
        truncated = False
        info = {"reason": reason}
        
        return state, reward, terminated, truncated, info

    def log_step(self, reason, terminal_reward=None):
        if terminal_reward is None:
            terminal_reward = self.account_information["reward"][-1]
            
        cash = self.account_information["cash"][-1]
        total_assets = self.account_information["total_assets"][-1]
        cash_pct = cash / total_assets if total_assets > 0 else 0
        gl_pct = total_assets / self.initial_amount
        
        rec = [
            self.episode,
            self.date_index - self.starting_point,
            reason,
            f"{self.currency}{cash:0,.0f}",
            f"{self.currency}{total_assets:0,.0f}",
            f"{terminal_reward*100:0.5f}%",
            f"{(gl_pct - 1)*100:0.5f}%",
            f"{cash_pct*100:0.2f}%",
        ]
        
        self.episode_history.append(rec)
        
        if not self.printed_header:
            self.log_header()
        if (self.current_step + 1) % self.print_verbosity == 0:
            print(self.template.format(*rec))

    def log_header(self):
        self.template = "{0:4}|{1:4}|{2:15}|{3:15}|{4:15}|{5:10}|{6:10}|{7:10}"
        print(
            self.template.format(
                "EPISODE", "STEPS", "TERMINAL_REASON", "CASH", "TOT_ASSETS",
                "TERMINAL_REWARD", "GAINLOSS_PCT", "CASH_PROPORTION"
            )
        )
        self.printed_header = True

    def get_reward(self):
        """Optimized reward calculation with vectorized operations"""
        if self.current_step == 0:
            return 0.0
            
        total_assets = self.account_information["total_assets"][-1]
        cash = self.account_information["cash"][-1]
        holdings = np.array(self.state_memory[-1][1:1+self.num_assets], dtype=np.float32)
        
        # Vectorized penalty calculations
        neg_closing_diff = np.minimum(self.closing_diff_avg_buy, 0.0)
        neg_profit_sell_diff = np.minimum(self.profit_sell_diff_avg_buy, 0.0)
        pos_profit_sell_diff = np.maximum(self.profit_sell_diff_avg_buy, 0.0)
        
        cash_penalty = max(0.0, (total_assets * self.cash_penalty_proportion - cash))
        
        if self.current_step > 1:
            prev_holdings = np.array(self.state_memory[-2][1:1+self.num_assets], dtype=np.float32)
            stop_loss_penalty = -np.dot(prev_holdings, neg_closing_diff)
        else:
            stop_loss_penalty = 0.0
            
        low_profit_penalty = -np.dot(holdings, neg_profit_sell_diff)
        additional_reward = np.dot(holdings, pos_profit_sell_diff)
        
        # Normalize
        cash_penalty /= self.initial_amount
        stop_loss_penalty /= self.initial_amount
        low_profit_penalty /= self.initial_amount
        additional_reward /= self.initial_amount
        
        total_penalty = cash_penalty + stop_loss_penalty + low_profit_penalty
        
        reward = ((total_assets - total_penalty + additional_reward) / 
                 self.initial_amount - 1.0)
        
        return float(reward)

    def step(self, actions):
        # Convert to numpy array once
        actions = np.asarray(actions, dtype=np.float32).flatten()
        current_state = self.state_memory[-1]
        holdings = current_state[1:1+self.num_assets].copy()
        
        # Fast close prices access
        closings = self.close_prices[self.date_index]
        
        # Track trades
        self.sum_trades += np.sum(np.abs(actions))
        
        # Check termination
        if self.date_index >= self.num_dates - 1:
            return self.return_terminal(reward=self.get_reward())
        
        # Calculate portfolio value
        begin_cash = current_state[0]
        asset_value = np.dot(holdings, closings)
        reward = self.get_reward()
        
        # Store account info
        self.account_information["cash"].append(begin_cash)
        self.account_information["asset_value"].append(asset_value)
        self.account_information["total_assets"].append(begin_cash + asset_value)
        self.account_information["reward"].append(reward)
        
        # Scale actions
        actions_scaled = actions * self.hmax
        self.actions_memory.append(actions_scaled * closings)
        
        # Convert to shares
        valid_prices = closings > 1e-8
        shares = np.where(valid_prices, actions_scaled / closings, 0.0)
        
        # Apply turbulence stop if needed
        if (self.turbulence_threshold is not None and 
            self.turbulence >= self.turbulence_threshold):
            shares = -holdings
        
        # Clip to available holdings
        shares = np.clip(shares, -holdings, np.inf)
        
        # Stop-loss mechanism
        self.closing_diff_avg_buy = closings - (self.stoploss_penalty * self.avg_buy_price)
        stop_loss_mask = (begin_cash >= self.stoploss_penalty * self.initial_amount) & (self.closing_diff_avg_buy < 0)
        shares = np.where(stop_loss_mask, -holdings, shares)
        
        # Calculate transactions
        sells = np.maximum(-shares, 0)
        buys = np.maximum(shares, 0)
        
        proceeds = np.dot(sells, closings)
        spend = np.dot(buys, closings)
        costs = proceeds * self.sell_cost_pct + spend * self.buy_cost_pct
        coh = begin_cash + proceeds
        
        # Handle cash shortages
        if (spend + costs) > coh:
            if self.patient:
                shares = np.minimum(shares, 0)
                sells = np.maximum(-shares, 0)
                proceeds = np.dot(sells, closings)
                costs = proceeds * self.sell_cost_pct
                spend = 0
            else:
                return self.return_terminal(reason="CASH SHORTAGE", reward=self.get_reward())
        
        # Update cash and holdings
        coh = coh - spend - costs
        holdings_updated = holdings + shares
        
        # Update average buy price (vectorized)
        buy_mask = shares > 1e-8
        self.n_buys += buy_mask.astype(np.float32)
        
        valid_updates = buy_mask & (self.n_buys > 0)
        price_diffs = closings - self.avg_buy_price
        self.avg_buy_price = np.where(
            valid_updates,
            self.avg_buy_price + (price_diffs / self.n_buys),
            self.avg_buy_price
        )
        
        # Reset for zero holdings
        zero_holdings = holdings_updated <= 1e-8
        self.n_buys[zero_holdings] = 0
        self.avg_buy_price[zero_holdings] = 0
        
        # Profit tracking
        sell_mask = sells > 1e-8
        profit_sell = (closings - self.avg_buy_price) > 0
        self.profit_sell_diff_avg_buy = np.where(
            sell_mask & profit_sell,
            closings - (self.min_profit_penalty * self.avg_buy_price),
            0.0
        )
        
        # Move to next time step
        self.date_index += 1
        self.actual_num_trades = np.sum(np.abs(shares) > 1e-8)
        
        # Create new state
        new_state = np.zeros(self.state_space, dtype=np.float32)
        new_state[0] = coh
        new_state[1:1+self.num_assets] = holdings_updated
        new_state[1+self.num_assets:] = self.state_vectors[self.date_index]
        
        self.state_memory.append(new_state)
        self.transaction_memory.append(shares)
        
        return new_state, reward, False, False, {}

    def get_sb_env(self):
        def get_self():
            return deepcopy(self)
        e = DummyVecEnv([get_self])
        obs = e.reset()
        return e, obs

    def get_multiproc_env(self, n=10):
        def make_env(rank):
            def _init():
                env = deepcopy(self)
                env = Monitor(env)
                return env
            return _init
        e = SubprocVecEnv([make_env(i) for i in range(n)], start_method="fork")
        obs = e.reset()
        return e, obs

    def save_asset_memory(self):
        if len(self.account_information["cash"]) == 0:
            return None
        dates_used = self.dates[self.starting_point:self.starting_point + len(self.account_information["cash"])]
        account_df = pd.DataFrame(self.account_information)
        account_df["date"] = dates_used
        return account_df

    def save_action_memory(self):
        if len(self.actions_memory) == 0:
            return None
        dates_used = self.dates[self.starting_point:self.starting_point + len(self.actions_memory)]
        return pd.DataFrame({
            "date": dates_used,
            "actions": self.actions_memory,
            "transactions": self.transaction_memory,
        })
