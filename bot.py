
import telebot
from strategy_engine import get_crypto_price, get_forex_rate

bot = telebot.TeleBot("8111192844:AAHuVZYs6RolBhdqPpTWW9g7ksGRaq3p0WA")

@bot.message_handler(commands=["start", "help"])
def send_welcome(message):
    bot.reply_to(message, "سلام! برای دریافت قیمت ارز دیجیتال یا جفت‌ارز، دستور زیر را وارد کن:\n\n"
                          "/crypto BTCUSDT\n"
                          "/forex USD EUR")

@bot.message_handler(commands=["crypto"])
def crypto_price(message):
    try:
        _, symbol = message.text.split()
        price = get_crypto_price(symbol)
        if price:
            bot.reply_to(message, f"قیمت لحظه‌ای {symbol.upper()}: {price}")
        else:
            bot.reply_to(message, "نماد وارد شده صحیح نیست یا مشکلی در دریافت داده‌ها وجود دارد.")
    except:
        bot.reply_to(message, "لطفا نماد را به صورت صحیح وارد کنید. مثال: /crypto BTCUSDT")

@bot.message_handler(commands=["forex"])
def forex_price(message):
    try:
        _, base, target = message.text.split()
        rate = get_forex_rate(base, target)
        if rate:
            bot.reply_to(message, f"نرخ تبدیل {base.upper()} به {target.upper()}: {rate}")
        else:
            bot.reply_to(message, "نماد وارد شده صحیح نیست یا مشکلی در دریافت داده‌ها وجود دارد.")
    except:
        bot.reply_to(message, "لطفا نمادها را به صورت صحیح وارد کنید. مثال: /forex USD EUR")

bot.polling()
