import os
from datetime import datetime

from flask import Flask, abort, request

# https://github.com/line/line-bot-sdk-python
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

app = Flask(__name__)

line_bot_api = LineBotApi('ArjtHTRBrCzfbKcIVIgaZI6iNVVGduEgSTcCe7WRizbMtZL7KP6XG02U/q/fKDQRK+ffSjGH/wPwV5ctoEdWDM2F/vaxaVjq2gd1AdpWdTDxVaZUxP5ft7ApKPSiRJVvvXUOemyIRWN2ryxJBsZmkAdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('dc64c782fed6cb4e9f7fc02482438aac')

@app.route("/", methods=["GET", "POST"])
def callback():

    if request.method == "GET":
        return "Hello Heroku"
    if request.method == "POST":
        signature = request.headers["X-Line-Signature"]
        body = request.get_data(as_text=True)

        try:
            handler.handle(body, signature)
        except InvalidSignatureError:
            abort(400)

        return "OK"


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    get_message = event.message.text

    # Send To Line
    reply = TextSendMessage(text=f"{get_message}")
    line_bot_api.reply_message(event.reply_token, reply)
