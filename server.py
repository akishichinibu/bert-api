from flask import Flask

from module.bert import bp as bert_bp

def create_app():
    app = Flask(__name__, static_url_path='')
    app.register_blueprint(bert_bp, url_prefix='/api')
    return app

app = create_app()
