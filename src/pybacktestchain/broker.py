import pandas as pd
import logging
from dataclasses import dataclass
from datetime import datetime

import os 
import pickle
from src.pybacktestchain.data_module import UNIVERSE_SEC, FirstTwoMoments, get_stocks_data, DataModule, Information
from src.pybacktestchain.utils import generate_random_name
from src.pybacktestchain.blockchain import Block, Blockchain
from numba import jit 

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datetime import timedelta, datetime

#---------------------------------------------------------
# Classes
#---------------------------------------------------------

@dataclass
class Position:
    ticker: str
    quantity: int
    entry_price: float

@dataclass
class Broker:
    cash: float 
    positions: dict = None
    transaction_log: pd.DataFrame = None
    entry_prices: dict = None
    verbose: bool = True

    def initialize_blockchain(self, name: str):
        # Check if the blockchain is already initialized and stored in the blockchain folder
        # if folder blockchain does not exist, create it
        if not os.path.exists('blockchain'):
            os.makedirs('blockchain')
        chains = os.listdir('blockchain')
        ending = f'{name}.pkl'
        if ending in chains:
            if self.verbose:
                logging.warning(f"Blockchain with name {name} already exists. Please use a different name.")
            with open(f'blockchain/{name}.pkl', 'rb') as f:
                self.blockchain = pickle.load(f)
            return

        self.blockchain = Blockchain(name)
        # Store the blockchain
        self.blockchain.store()

        if self.verbose:
            logging.info(f"Blockchain with name {name} initialized and stored in the blockchain folder.")

    def __post_init__(self):
        # Initialize positions as a dictionary of Position objects
        if self.positions is None:
            self.positions = {}
        # Initialize the transaction log as an empty DataFrame if none is provided
        if self.transaction_log is None:
            self.transaction_log = pd.DataFrame(columns=['Date', 'Action', 'Ticker', 'Quantity', 'Price', 'Cash'])
    

        # Initialize the entry prices as a dictionary
        if self.entry_prices is None:
            self.entry_prices = {}

    def buy(self, ticker: str, quantity: int, price: float, date: datetime):
        """Executes a buy order for the specified ticker."""
        total_cost = price * quantity
        if self.cash >= total_cost:
            self.cash -= total_cost
            if ticker in self.positions:
                position = self.positions[ticker]
                new_quantity = position.quantity + quantity
                new_entry_price = ((position.entry_price * position.quantity) + (price * quantity)) / new_quantity
                position.quantity = new_quantity
                position.entry_price = new_entry_price
            else:

                self.positions[ticker] = Position(ticker, quantity, price)
            self.log_transaction(date, 'BUY', ticker, quantity, price)
            self.entry_prices[ticker] = price
        else:
            if self.verbose:
                logging.warning(f"Not enough cash to buy {quantity} shares of {ticker} at {price}. Available cash: {self.cash}")
    
    def sell(self, ticker: str, quantity: int, price: float, date: datetime):
        """Executes a sell order for the specified ticker."""
        if ticker in self.positions and self.positions[ticker].quantity >= quantity:
            position = self.positions[ticker]
            position.quantity -= quantity
            self.cash += price * quantity

            if position.quantity == 0:
                del self.positions[ticker]
                del self.entry_prices[ticker]
            self.log_transaction(date, 'SELL', ticker, quantity, price)
        else:
            if self.verbose:
                logging.warning(f"Not enough shares to sell {quantity} shares of {ticker}. Position size: {self.positions.get(ticker, 0)}")
    
    def log_transaction(self, date, action, ticker, quantity, price):
        """Logs the transaction."""
        transaction = pd.DataFrame([{
            'Date': date,
            'Action': action,
            'Ticker': ticker,
            'Quantity': quantity,
            'Price': price,
            'Cash': self.cash
        }])

        self.transaction_log = pd.concat([self.transaction_log, transaction], ignore_index=True)

    def get_cash_balance(self):
        return self.cash

    def get_transaction_log(self):
        return self.transaction_log

    def get_portfolio_value(self, market_prices: dict):
        """Calculates the total portfolio value based on the current market prices."""
        portfolio_value = self.cash
        for ticker, position in self.positions.items():
            portfolio_value += position.quantity * market_prices[ticker]
        return portfolio_value
    
    def execute_portfolio(self, portfolio: dict, prices: dict, date: datetime):
        """Executes the trades for the portfolio based on the generated weights."""
        
        # First, handle all the sell orders to free up cash
        for ticker, weight in portfolio.items():
            price = prices.get(ticker)
            if price is None:
                if self.verbose:
                    logging.warning(f"Price for {ticker} not available on {date}")
                continue
            
            total_value = self.get_portfolio_value(prices)
            target_value = total_value * weight
            current_value = self.positions.get(ticker, Position(ticker, 0, 0)).quantity * price
            diff_value = target_value - current_value
            quantity_to_trade = int(diff_value / price)
            
            if quantity_to_trade < 0:
                self.sell(ticker, abs(quantity_to_trade), price, date)
        
        # Then, handle all the buy orders, checking if there's enough cash
        for ticker, weight in portfolio.items():
            price = prices.get(ticker)
            if price is None:
                if self.verbose:
                    logging.warning(f"Price for {ticker} not available on {date}")
                continue
      
            total_value = self.get_portfolio_value(prices)
            target_value = total_value * weight
            current_value = self.positions.get(ticker, Position(ticker, 0, 0)).quantity * price
            diff_value = target_value - current_value
            quantity_to_trade = int(diff_value / price)
           
            if quantity_to_trade > 0:
                available_cash = self.get_cash_balance()
                cost = quantity_to_trade * price
                
                if cost <= available_cash:
                    self.buy(ticker, quantity_to_trade, price, date)
                else:
                    if self.verbose:
                        logging.warning(f"Not enough cash to buy {quantity_to_trade} of {ticker} on {date}. Needed: {cost}, Available: {available_cash}")
                        logging.info(f"Buying as many shares of {ticker} as possible with available cash.")
                    quantity_to_trade = int(available_cash / price)
                    self.buy(ticker, quantity_to_trade, price, date)

    def get_transaction_log(self):
        """Returns the transaction log."""
        return self.transaction_log

@dataclass
class RebalanceFlag:
    def time_to_rebalance(self, t: datetime):
        pass 

# Implementation of e.g. rebalancing at the end of each month
@dataclass
class EndOfMonth(RebalanceFlag):
    def time_to_rebalance(self, t: datetime):
        # Convert to pandas Timestamp for convenience
        pd_date = pd.Timestamp(t)
        # Get the last business day of the month
        last_business_day = pd_date + pd.offsets.BMonthEnd(0)
        # Check if the given date matches the last business day
        return pd_date == last_business_day

@dataclass
class RiskModel:
    def trigger_stop_loss(self, t: datetime, portfolio: dict, prices: dict):
        pass

@dataclass
class StopLoss(RiskModel):
    threshold: float = 0.1
    def trigger_stop_loss(self, t: datetime, portfolio: dict, prices: dict, broker: Broker):
        
        for ticker, position in list(broker.positions.items()):
            entry_price = broker.entry_prices[ticker]
            current_price = prices.get(ticker)
            if current_price is None:
                logging.warning(f"Price for {ticker} not available on {t}")
                continue
            # Calculate the loss percentage
            loss = (current_price - entry_price) / entry_price
            if loss < -self.threshold:
                logging.info(f"Stop loss triggered for {ticker} at {t}. Selling all shares.")
                broker.sell(ticker, position.quantity, current_price, t)
@dataclass
class Backtest:
    initial_date: datetime
    final_date: datetime
    universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'INTC', 'CSCO', 'NFLX']
    information_class : type  = Information
    s: timedelta = timedelta(days=360)
    time_column: str = 'Date'
    company_column: str = 'ticker'
    adj_close_column : str ='Adj Close'
    rebalance_flag : type = EndOfMonth
    risk_model : type = StopLoss
    initial_cash: int = 1000000  # Initial cash in the portfolio
    name_blockchain: str = 'backtest'
    verbose: bool = True
    broker = Broker(cash=initial_cash, verbose=verbose)
    
    def __post_init__(self):
        self.backtest_name = generate_random_name()
        self.broker.initialize_blockchain(self.name_blockchain)

    def run_backtest(self):
        logging.info(f"Running backtest from {self.initial_date} to {self.final_date}.")
        logging.info(f"Retrieving price data for universe")
        self.risk_model = self.risk_model(threshold=0.1)
        # self.initial_date to yyyy-mm-dd format
        init_ = self.initial_date.strftime('%Y-%m-%d')
        # self.final_date to yyyy-mm-dd format
        final_ = self.final_date.strftime('%Y-%m-%d')
        df = get_stocks_data(self.universe, init_, final_)

        # Initialize the DataModule
        data_module = DataModule(df)

        # Create the Information object
        info = self.information_class(s = self.s, 
                                    data_module = data_module,
                                    time_column=self.time_column,
                                    company_column=self.company_column,
                                    adj_close_column=self.adj_close_column)
        
        # Run the backtest
        for t in pd.date_range(start=self.initial_date, end=self.final_date, freq='D'):
            
            if self.risk_model is not None:
                portfolio = info.compute_portfolio(t, info.compute_information(t))
                prices = info.get_prices(t)
                self.risk_model.trigger_stop_loss(t, portfolio, prices, self.broker)
           
            if self.rebalance_flag().time_to_rebalance(t):
                logging.info("-----------------------------------")
                logging.info(f"Rebalancing portfolio at {t}")
                information_set = info.compute_information(t)
                portfolio = info.compute_portfolio(t, information_set)
                prices = info.get_prices(t)
                self.broker.execute_portfolio(portfolio, prices, t)

        logging.info(f"Backtest completed. Final portfolio value: {self.broker.get_portfolio_value(info.get_prices(self.final_date))}")
        df = self.broker.get_transaction_log()

        # create backtests folder if it does not exist
        if not os.path.exists('backtests'):
            os.makedirs('backtests')

        # save to csv, use the backtest name 
        df.to_csv(f"backtests/{self.backtest_name}.csv")

        # store the backtest in the blockchain
        self.broker.blockchain.add_block(self.backtest_name, df.to_string())
    
