from telegram import Bot
from telegram.ext import Updater, CommandHandler
from strategy_engine import generate_crypto_signals, generate_forex_signals

TOKEN = '8111192844:AAHuVZYs6RolBhdqPpTWW9g7ksGRaq3p0WA'
CHAT_ID = 632886964

def start(update, context):
    update.message.reply_text('ربات سیگنال فعال شد.')

def send_signals(context):
    crypto_signals = generate_crypto_signals()
    forex_signals = generate_forex_signals()

    for signal in crypto_signals + forex_signals:
        market = 'CRYPTO' if 'USDT' in signal['symbol'] else 'FOREX'
        msg = (
            'New Signal (' + market + ')\n'
            'Symbol: ' + str(signal['symbol']) + '\n'
            'Timeframe: ' + str(signal['tf']) + '\n'
            'Entry Price: ' + str(signal['entry']) + '\n'
            'Take Profit (TP): ' + str(signal['tp']) + '\n'
            'Stop Loss (SL): ' + str(signal['sl']) + '\n'
            'Confidence: ' + str(signal['confidence']) + '%\n'
            'Volatility: ' + str(signal['volatility']) + '%\n'
            'Analysis: ' + str(signal['analysis'])
        )
        context.bot.send_message(chat_id=CHAT_ID, text=msg)

updater = Updater(TOKEN, use_context=True)
dp = updater.dispatcher
dp.add_handler(CommandHandler('start', start))
updater.job_queue.run_repeating(send_signals, interval=900, first=10)
updater.start_polling()
updater.idle()
