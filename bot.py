import asyncio

from telegram import Bot

from strategy_engine import generate_signals



API_TOKEN = "8111192844:AAHuVZYs6RolBhdqPpTWW9g7ksGRaq3p0WA"

USER_ID = 632886964  # ���� ���� ����� ���



bot = Bot(token=API_TOKEN)



async def main():

    while True:

        signals = generate_signals()

        for sig in signals:

            message = f"""

?? ����� ����:

�����: {sig['symbol']}

����: {sig['entry']}

�� ���: {sig['tp']}

�� ���: {sig['sl']}

�������: {sig['confidence']}?

�����: {sig['analysis']}

            """

            await bot.send_message(chat_id=USER_ID, text=message)

        await asyncio.sleep(300)  # �� ? �����



if __name__ == "__main__":

    asyncio.run(main())

