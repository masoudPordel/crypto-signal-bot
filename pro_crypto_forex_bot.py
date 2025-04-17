
import ccxt
import pandas as pd
import pandas_ta as ta
from telegram import Bot
from telegram.ext import Updater, JobQueue

TELEGRAM_TOKEN = '8111192844:AAHuVZYs6RolBhdqPpTWW9g7ksGRaq3p0WA'
CHAT_ID = '@MasoudPordel'

# ارزهای دیجیتال و جفت‌ارزها
PAIRS = {
    'crypto': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
    'forex': ['EUR/USD', 'GBP/USD', 'USD/JPY']
}

# تنظیمات اکسچنج
exchange = ccxt.binance({
    'enableRateLimit': True
})

def calculate_signals(pair, timeframe='5m'):
    try:
        ohlcv = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=150)
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])

        # اندیکاتورها
        df['ema20'] = ta.ema(df['close'], length=20)
        df['ema50'] = ta.ema(df['close'], length=50)
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['stochrsi'] = ta.stochrsi(df['close']).iloc[:,0]
        macd = ta.macd(df['close'])
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['adx'] = ta.adx(df['high'], df['low'], df['close'])['ADX_14']
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'])
        df['willr'] = ta.willr(df['high'], df['low'], df['close'])

        # سیگنال خرید/فروش ترکیبی
        last = df.iloc[-1]
        previous = df.iloc[-2]

        buy_signals = []
        sell_signals = []

        # استراتژی خرید
        if (
            last['ema20'] > last['ema50'] and
            previous['ema20'] <= previous['ema50'] and
            last['rsi'] > 50 and last['stochrsi'] < 20 and
            last['macd'] > last['macd_signal'] and
            last['adx'] > 20 and last['mfi'] < 80 and
            last['willr'] < -80
        ):
            buy_signals.append('سیگنال خرید ترکیبی صادر شد!')

        # استراتژی فروش
        if (
            last['ema20'] < last['ema50'] and
            previous['ema20'] >= previous['ema50'] and
            last['rsi'] < 50 and last['stochrsi'] > 80 and
            last['macd'] < last['macd_signal'] and
            last['adx'] > 20 and last['mfi'] > 20 and
            last['willr'] > -20
        ):
            sell_signals.append('سیگنال فروش ترکیبی صادر شد!')

        result = ""
        if buy_signals:
            tp = round(last['close'] * 1.02, 2)
            sl = round(last['close'] * 0.98, 2)
            result += f"**{pair}**:
" + "\n".join(buy_signals)
            result += f"\n⤴️ حد سود: `{tp}`\n⤵️ حد ضرر: `{sl}`"

        elif sell_signals:
            tp = round(last['close'] * 0.98, 2)
            sl = round(last['close'] * 1.02, 2)
            result += f"**{pair}**:
" + "\n".join(sell_signals)
            result += f"\n⤵️ حد سود: `{tp}`\n⤴️ حد ضرر: `{sl}`"

        return result if result else None
    except Exception as e:
        return f"⚠️ خطا در {pair}: {e}"

def send_all_signals(context):
    for market, pairs in PAIRS.items():
        for pair in pairs:
            signal = calculate_signals(pair)
            if signal:
                context.bot.send_message(chat_id=CHAT_ID, text=signal, parse_mode='Markdown')

def main():
    updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
    job_queue = updater.job_queue
    job_queue.run_repeating(send_all_signals, interval=300, first=10)
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
