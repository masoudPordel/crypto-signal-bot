import asyncio
import telegram
import logging
from strategy_engine import analyze
import pandas as pd
import random

BOT_TOKEN = "8111192844:AAHuVZYs6RolBhdqPpTWW9g7ksGRaq3p0WA"
CHAT_ID = 632886964

logging.basicConfig(level=logging.INFO)

async def generate_fake_data():
    ts = pd.date_range(end=pd.Timestamp.now(), periods=100, freq="1H")
    df = pd.DataFrame(index=ts)
    df["open"] = np.random.rand(len(df)) * 100
    df["high"] = df["open"] + np.random.rand(len(df)) * 5
    df["low"] = df["open"] - np.random.rand(len(df)) * 5
    df["close"] = df["open"] + np.random.randn(len(df))
    df["volume"] = np.random.randint(1000, 10000, size=len(df))
    return df

async def send_signals():
    await bot.send_message(chat_id=CHAT_ID, text="ربات آماده ارسال سیگنال است.")
    df = await generate_fake_data()
    signal = analyze(df)

    if signal:
        msg = f"""📢 سیگنال {"خرید" if float(signal["هدف سود"]) > float(signal["قیمت ورود"]) else "فروش"}

نماد: {signal['نماد']}
تایم‌فریم: {signal['تایم‌فریم']}
قیمت ورود: {signal['قیمت ورود']}
هدف سود: {signal['هدف سود']}
حد ضرر: {signal['حد ضرر']}
سطح اطمینان: {signal['سطح اطمینان']}%
ریسک به ریوارد: {signal['ریسک به ریوارد']}

تحلیل:
{signal['تحلیل']}
"""
        await bot.send_message(chat_id=CHAT_ID, text=msg)

if __name__ == "__main__":
    asyncio.run(send_signals())