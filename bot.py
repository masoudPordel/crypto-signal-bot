import asyncio

from telegram import Bot

from strategy_engine import generate_signals



API_TOKEN = "8111192844:AAHuVZYs6RolBhdqPpTWW9g7ksGRaq3p0WA"

USER_ID = 632886964  # ÂíÏí ÚÏÏí ÊáÑÇã ÔãÇ



bot = Bot(token=API_TOKEN)



async def main():

    while True:

        signals = generate_signals()

        for sig in signals:

            message = f"""

?? ÓíäÇá ÌÏíÏ:

ÓíãÈá: {sig['symbol']}

æÑæÏ: {sig['entry']}

ÍÏ ÓæÏ: {sig['tp']}

ÍÏ ÖÑÑ: {sig['sl']}

ÇØãíäÇä: {sig['confidence']}?

ÊÍáíá: {sig['analysis']}

            """

            await bot.send_message(chat_id=USER_ID, text=message)

        await asyncio.sleep(300)  # åÑ ? ÏŞíŞå



if __name__ == "__main__":

    asyncio.run(main())

