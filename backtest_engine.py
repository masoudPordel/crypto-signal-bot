
import backtrader as bt

class SignalStrategy(bt.Strategy):
    def __init__(self):
        self.rsi = bt.ind.RSI(period=14)

    def next(self):
        if self.rsi < 30:
            self.buy()
        elif self.rsi > 70:
            self.sell()

def run_backtest(df):
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(SignalStrategy)
    cerebro.broker.setcash(10000)
    cerebro.run()
    return cerebro.broker.getvalue()
