import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.callbacks import BaseCallbackHandler


class LiveTokenHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="", flush=True)

    def on_llm_end(self, response, **kwargs) -> None:
        print()  # move to next line after response finishes


api_key = "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"

os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

system_message = SystemMessage(
    content=(
        "You are a translation assistant. "
        "Translate the user's text into Kinyarwanda only. "
        "Return only the translation, with no explanation."
    )
)

llm = ChatOpenAI(
    model="google/gemini-2.5-flash-lite",   # fast choice
    temperature=0,
    max_tokens=80, # type: ignore
    streaming=True,
    callbacks=[LiveTokenHandler()],
)

while True:
    text = input("\nYou: ").strip()

    if text.lower() in ["exit", "quit"]:
        print("Stopped.")
        break

    print("Bot: ", end="", flush=True)

    llm.invoke([
        system_message,
        HumanMessage(content=text)
    ])