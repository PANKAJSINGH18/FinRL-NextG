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
    OPTIMIZED VERSION:
    - Precomputed data for faster access
    - Vectorized operations
    - Reduced pandas usage
    - Memory optimizations
    - Cached calculations
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
        print_verbosity=10,
        initial_amount=1e6,
        daily_information_cols=["open", "close", "high", "low", "volume"],
        cache_indicator_data=True,
        cash_penalty_proportion=0.1,
        random_start=True,
        patient=False,
        currency="$",
    ):
        self.df = df
        self.stock_col = "tic"
        self.assets = df[self.stock_col].unique()
        self.dates = df[date_col_name].sort_values().unique()
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
        self.state_space = (
            1 + len(self.assets) + len(self.assets) * len(self.daily_information_cols)
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.assets),))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.cash_penalty_proportion = cash_penalty_proportion
        
        # ===== OPTIMIZATION 1: Precompute all data =====
        self._precompute_all_data(df, date_col_name)
        
        # ===== OPTIMIZATION 2: Pre-calculate constants =====
        self.num_assets = len(self.assets)
        self.asset_indices = np.arange(self.num_assets)
        self.holdings_slice = slice(1, 1 + self.num_assets)
        
        self.turbulence = 0
        self.episode = -1
        self.episode_history = []
        self.printed_header = False
        
        # ===== OPTIMIZATION 3: Reusable arrays =====
        self._reusable_arrays = {
            'zeros_assets': np.zeros(self.num_assets, dtype=np.float32),
            'ones_assets': np.ones(self.num_assets, dtype=np.float32),
        }

    def _precompute_all_data(self, df, date_col_name):
        """Precompute all environment data for maximum performance"""
        print("🔄 Precomputing environment data...")
        start_time = time.time()
        
        # Convert to numpy arrays for faster access
        self.df_numpy = {}
        self.date_to_index = {}
        
        # Create fast lookup structure
        unique_dates = df[date_col_name].sort_values().unique()
        self.dates = unique_dates
        
        # Precompute all date vectors
        self.precomputed_date_vectors = []
        df_indexed = df.set_index(date_col_name)
        
        for i, date in enumerate(unique_dates):
            date_str = str(date)
            date_data = []
            
            for asset in self.assets:
                asset_data = df_indexed.loc[date_str]
                if len(asset_data.shape) == 1:  # Single row
                    asset_data = asset_data.to_frame().T
                
                asset_subset = asset_data[asset_data[self.stock_col] == asset]
                if len(asset_subset) > 0:
                    row = asset_subset.iloc[0]
                    date_data.extend([row[col] for col in self.daily_information_cols])
                else:
                    # Handle missing data with zeros
                    date_data.extend([0.0] * len(self.daily_information_cols))
            
            self.precomputed_date_vectors.append(np.array(date_data, dtype=np.float32))
            self.date_to_index[date_str] = i
        
        self.precomputed_date_vectors = np.array(self.precomputed_date_vectors)
        
        # Precompute close prices for fast access
        self.close_prices = self.precomputed_date_vectors[:, 
                          [i * len(self.daily_information_cols) + 
                           self.daily_information_cols.index('close') 
                           for i in range(len(self.assets))]]
        
        print(f"✅ Data precomputed in {time.time() - start_time:.2f}s")
        print(f"   - Dates: {len(self.dates)}")
        print(f"   - Assets: {len(self.assets)}")
        print(f"   - State size: {self.state_space}")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def current_step(self):
        return self.date_index - self.starting_point

    def reset(self, *, seed=None, options=None):
        self.seed()
        
        # ===== OPTIMIZATION 4: Reuse arrays =====
        self.sum_trades = 0
        self.actual_num_trades = 0
        self.closing_diff_avg_buy = self._reusable_arrays['zeros_assets'].copy()
        self.profit_sell_diff_avg_buy = self._reusable_arrays['zeros_assets'].copy()
        self.n_buys = self._reusable_arrays['zeros_assets'].copy()
        self.avg_buy_price = self._reusable_arrays['zeros_assets'].copy()
        
        if self.random_start:
            starting_point = random.choice(range(int(len(self.dates) * 0.5)))
            self.starting_point = starting_point
        else:
            self.starting_point = 0
            
        self.date_index = self.starting_point
        self.turbulence = 0
        self.episode += 1
        
        # ===== OPTIMIZATION 5: Pre-allocate memory =====
        self.actions_memory = []
        self.transaction_memory = []
        self.state_memory = []
        
        self.account_information = {
            "cash": [],
            "asset_value": [],
            "total_assets": [],
            "reward": [],
        }
        
        # ===== OPTIMIZATION 6: Fast state initialization =====
        init_state = np.zeros(self.state_space, dtype=np.float32)
        init_state[0] = self.initial_amount  # cash
        # holdings are already zeros (positions 1:1+num_assets)
        # Add precomputed date vector
        init_state[1 + self.num_assets:] = self.precomputed_date_vectors[self.date_index]
        
        self.state_memory.append(init_state)
        
        return init_state, {}

    def get_date_vector(self, date_index, cols=None):
        """Optimized date vector access"""
        return self.precomputed_date_vectors[date_index].copy()

    def get_close_prices(self, date_index):
        """Fast close prices access"""
        return self.close_prices[date_index].copy()

    def return_terminal(self, reason="Last Date", reward=0):
        state = self.state_memory[-1]
        self.log_step(reason=reason, terminal_reward=reward)
        
        # Log metrics
        gl_pct = self.account_information["total_assets"][-1] / self.initial_amount
        logger.info("environment/GainLoss_pct: %.2f", (gl_pct - 1) * 100)
        logger.info("environment/total_assets: %d", int(self.account_information["total_assets"][-1]))
        
        reward_pct = self.account_information["total_assets"][-1] / self.initial_amount
        logger.info("environment/total_reward_pct: %.2f", (reward_pct - 1) * 100)
        logger.info("environment/total_trades: %d", self.sum_trades)
        logger.info("environment/actual_num_trades: %d", self.actual_num_trades)
        
        if self.current_step > 0:
            logger.info("environment/avg_daily_trades: %.2f", self.sum_trades / self.current_step)
            logger.info("environment/avg_daily_trades_per_asset: %.2f", 
                       self.sum_trades / self.current_step / len(self.assets))
        
        logger.info("environment/completed_steps: %d", self.current_step)
        logger.info("environment/sum_rewards: %.4f", np.sum(self.account_information["reward"]))
        
        if self.account_information["total_assets"][-1] > 0:
            cash_prop = (self.account_information["cash"][-1] / 
                        self.account_information["total_assets"][-1])
            logger.info("environment/cash_proportion: %.4f", cash_prop)
        
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
        print(self.template.format(*rec))

    def log_header(self):
        self.template = "{0:4}|{1:4}|{2:15}|{3:15}|{4:15}|{5:10}|{6:10}|{7:10}"
        print(
            self.template.format(
                "EPISODE",
                "STEPS",
                "TERMINAL_REASON",
                "CASH",
                "TOT_ASSETS",
                "TERMINAL_REWARD",
                "GAINLOSS_PCT",
                "CASH_PROPORTION",
            )
        )
        self.printed_header = True

    def get_reward(self):
        """Optimized reward calculation"""
        if self.current_step == 0:
            return 0.0
            
        total_assets = self.account_information["total_assets"][-1]
        cash = self.account_information["cash"][-1]
        
        # ===== OPTIMIZATION 7: Vectorized operations =====
        holdings = np.array(self.state_memory[-1][self.holdings_slice], dtype=np.float32)
        
        # Vectorized clipping operations
        neg_closing_diff = np.minimum(self.closing_diff_avg_buy, 0.0)
        neg_profit_sell_diff = np.minimum(self.profit_sell_diff_avg_buy, 0.0)
        pos_profit_sell_diff = np.maximum(self.profit_sell_diff_avg_buy, 0.0)
        
        # Vectorized penalties
        cash_penalty = max(0.0, (total_assets * self.cash_penalty_proportion - cash))
        
        if self.current_step > 1:
            prev_holdings = np.array(self.state_memory[-2][self.holdings_slice], dtype=np.float32)
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
        # ===== OPTIMIZATION 8: Type consistency and vectorization =====
        actions = np.asarray(actions, dtype=np.float32).flatten()
        current_state = self.state_memory[-1]
        holdings = np.asarray(current_state[self.holdings_slice], dtype=np.float32)
        
        # Fast close prices access
        closings = self.get_close_prices(self.date_index)
        
        # Track trades
        self.sum_trades += np.sum(np.abs(actions))
        
        # Print header only once
        if not self.printed_header:
            self.log_header()
            
        # Check if we're at the end
        if self.date_index >= len(self.dates) - 1:
            return self.return_terminal(reward=self.get_reward())
        
        # ===== OPTIMIZATION 9: Batch calculations =====
        begin_cash = current_state[0]
        asset_value = np.dot(holdings, closings)
        reward = self.get_reward()
        
        # Store account information
        self.account_information["cash"].append(begin_cash)
        self.account_information["asset_value"].append(asset_value)
        self.account_information["total_assets"].append(begin_cash + asset_value)
        self.account_information["reward"].append(reward)
        
        # Scale actions
        actions = actions * self.hmax
        self.actions_memory.append(actions * closings)
        
        # Filter actions based on valid prices
        valid_actions_mask = closings > 1e-8  # Avoid floating point issues
        actions = np.where(valid_actions_mask, actions, 0.0)
        
        # Handle turbulence
        if (self.turbulence_threshold is not None and 
            self.turbulence >= self.turbulence_threshold):
            actions = -(holdings * closings)
        
        # Convert to shares if discrete
        if self.discrete_actions:
            shares = np.where(valid_actions_mask, actions / closings, 0.0)
            shares = shares.astype(np.int32)
            # Round to nearest shares_increment
            shares = np.where(
                shares >= 0,
                (shares // self.shares_increment) * self.shares_increment,
                ((shares + 1) // self.shares_increment) * self.shares_increment
            )
            actions = shares * closings
        else:
            shares = np.where(valid_actions_mask, actions / closings, 0.0)
        
        # Clip actions to prevent over-selling
        shares = np.maximum(shares, -holdings)
        actions = shares * closings
        
        # Stop-loss calculations
        self.closing_diff_avg_buy = closings - (self.stoploss_penalty * self.avg_buy_price)
        
        if begin_cash >= self.stoploss_penalty * self.initial_amount:
            stop_loss_mask = self.closing_diff_avg_buy < 0
            shares = np.where(stop_loss_mask, -holdings, shares)
            actions = shares * closings
        
        # Calculate transaction costs and proceeds
        sells = -np.minimum(shares, 0.0)
        proceeds = np.dot(sells, closings)
        costs = proceeds * self.sell_cost_pct
        
        buys = np.maximum(shares, 0.0)
        spend = np.dot(buys, closings)
        costs += spend * self.buy_cost_pct
        
        coh = begin_cash + proceeds
        
        # Handle cash shortages
        if (spend + costs) > coh:
            if self.patient:
                # Don't buy anything, only sell
                buy_mask = shares > 0
                shares = np.where(buy_mask, 0.0, shares)
                actions = shares * closings
                spend = 0.0
                costs = proceeds * self.sell_cost_pct
            else:
                return self.return_terminal(reason="CASH SHORTAGE", reward=self.get_reward())
        
        self.transaction_memory.append(shares)
        
        # Profit calculations
        sell_mask = sells > 0
        sell_closing_prices = np.where(sell_mask, closings, 0.0)
        profit_sell = (sell_closing_prices - self.avg_buy_price) > 0
        
        self.profit_sell_diff_avg_buy = np.where(
            profit_sell,
            closings - (self.min_profit_penalty * self.avg_buy_price),
            0.0
        )
        
        # Update holdings and cash
        coh = coh - spend - costs
        holdings_updated = holdings + shares
        
        # Update average buy price (vectorized)
        buy_mask = shares > 0
        self.n_buys += buy_mask.astype(np.float32)
        
        # Incremental average update
        price_diff = closings - self.avg_buy_price
        update_mask = buy_mask & (self.n_buys > 0)
        self.avg_buy_price = np.where(
            update_mask,
            self.avg_buy_price + (price_diff / self.n_buys),
            self.avg_buy_price
        )
        
        # Reset averages for zero holdings
        zero_holdings_mask = holdings_updated <= 1e-8
        self.n_buys = np.where(zero_holdings_mask, 0.0, self.n_buys)
        self.avg_buy_price = np.where(zero_holdings_mask, 0.0, self.avg_buy_price)
        
        # Update step
        self.date_index += 1
        self.actual_num_trades = np.sum(np.abs(np.sign(shares)))
        
        # ===== OPTIMIZATION 10: Fast state update =====
        new_state = np.zeros(self.state_space, dtype=np.float32)
        new_state[0] = coh
        new_state[self.holdings_slice] = holdings_updated
        new_state[1 + self.num_assets:] = self.precomputed_date_vectors[self.date_index]
        
        self.state_memory.append(new_state)
        
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
        if self.current_step == 0:
            return None
        dates_used = self.dates[self.starting_point:self.starting_point + len(self.account_information["cash"])]
        account_df = pd.DataFrame(self.account_information)
        account_df["date"] = dates_used
        return account_df

    def save_action_memory(self):
        if self.current_step == 0:
            return None
        dates_used = self.dates[self.starting_point:self.starting_point + len(self.actions_memory)]
        return pd.DataFrame({
            "date": dates_used,
            "actions": self.actions_memory,
            "transactions": self.transaction_memory,
        })
