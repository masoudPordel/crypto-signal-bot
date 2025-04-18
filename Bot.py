import asyncio
from telegram import Bot
from strategy_engine import generate_signals

API_TOKEN = "8111192844:AAHuVZYs6RolBhdqPpTWW9g7ksGRaq3p0WA"
USER_ID = 632886964  # Ø¢ÛØ¯Û Ø¹Ø¯Ø¯Û ØªÙÚ¯Ø±Ø§Ù Ø´ÙØ§

bot = Bot(token=API_TOKEN)

async def main():
    while True:
        signals = generate_signals()
        for sig in signals:
            message = f"""
ð Ø³ÛÚ¯ÙØ§Ù Ø¬Ø¯ÛØ¯:
Ø³ÛÙØ¨Ù: {sig['symbol']}
ÙØ±ÙØ¯: {sig['entry']}
Ø­Ø¯ Ø³ÙØ¯: {sig['tp']}
Ø­Ø¯ Ø¶Ø±Ø±: {sig['sl']}
Ø§Ø·ÙÛÙØ§Ù: {sig['confidence']}Ùª
ØªØ­ÙÛÙ: {sig['analysis']}
            """
            await bot.send_message(chat_id=USER_ID, text=message)
        await asyncio.sleep(300)  # ÙØ± Ûµ Ø¯ÙÛÙÙ

if __name__ == "__main__":
    asyncio.run(main())