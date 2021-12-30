from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)

app = Flask(__name__)

line_bot_api = LineBotApi('ArjtHTRBrCzfbKcIVIgaZI6iNVVGduEgSTcCe7WRizbMtZL7KP6XG02U/q/fKDQRK+ffSjGH/wPwV5ctoEdWDM2F/vaxaVjq2gd1AdpWdTDxVaZUxP5ft7ApKPSiRJVvvXUOemyIRWN2ryxJBsZmkAdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('dc64c782fed6cb4e9f7fc02482438aac')


@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    
    msg = (event.message.text).lower()
    
    if ('hello' in msg) or ('早安' in msg) or ('你好' in msg):
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="哈囉, 祝你有愉快的一天"))


if __name__ == "__main__":
    #line_bot_api.push_message(to, TextSendMessage(text='Hello World!'))

    app.run()