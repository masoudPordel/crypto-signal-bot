import schedule
import time
from strategy_engine import run_strategies
from telegram_utils import send_message

def job():
    print("در حال بررسی بازار...")
    signals = run_strategies()
    if signals:
        for sig in signals:
            send_message(sig)
    else:
        print("سیگنال جدیدی نبود.")

schedule.every(5).minutes.do(job)

while True:
    schedule.run_pending()
    time.sleep(1)