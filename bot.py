import os

import asyncio

import time

from telegram import Bot

from telegram.ext import Updater, CommandHandler

from telegram.utils.request import Request



from strategy_engine import generate_crypto_signals, generate_forex_signals



# تنظیم توکن و چت‌آیدی

TOKEN = os.getenv("BOT_TOKEN", "8111192844:AAHuVZYs6RolBhdqPpTWW9g7ksGRaq3p0WA")

CHAT_ID = int(os.getenv("CHAT_ID", "632886964"))



def start(update, context):

    update.message.reply_text("ربات سیگنال فعال شد. هر ۱۵ دقیقه یکبار سیگنال می‌آید.")



def send_signals(context):

    """این تابع توسط JobQueue هر ۱۵ دقیقه یک بار اجرا می‌شود."""

    bot = context.bot

    crypto = generate_crypto_signals()

    forex = generate_forex_signals()



    for sig in crypto + forex:

        try:

            market = "کریپتو" if "USDT" in sig["symbol"] else "فارکس"

            msg = (

                f"📡 سیگنال جدید ({market})\n\n"

                f"نماد: {sig['symbol']}\n"

                f"تایم‌فریم: {sig['tf']}\n"

                f"قیمت ورود: {sig['entry']}\n"

                f"حد سود (TP): {sig['tp']}\n"

                f"حد ضرر (SL): {sig['sl']}\n"

                f"درصد اطمینان: {sig['confidence']}%\n"

                f"نوسان: {sig['volatility']}%\n"

                f"تحلیل: {sig['analysis']}"

            )

            bot.send_message(chat_id=CHAT_ID, text=msg)

        except Exception as e:

            print(f"Error sending signal for {sig['symbol']}: {e}")



def main():

    request = Request(con_pool_size=8)

    bot = Bot(token=TOKEN, request=request)

    updater = Updater(bot=bot, use_context=True)

    dp = updater.dispatcher



    dp.add_handler(CommandHandler("start", start))

    updater.job_queue.run_repeating(send_signals, interval=900, first=10)



    # تأخیر برای جلوگیری از تداخل اتصالات

    print("ربات در حال شروع... 5 ثانیه صبر برای بستن اتصالات قدیمی.")

    time.sleep(5)



    try:

        updater.start_polling()

        print("ربات فعال شد...")

        updater.idle()

    except Exception as e:

        print(f"خطا در شروع ربات: {e}")

        updater.stop()



if __name__ == "__main__":

    main()

