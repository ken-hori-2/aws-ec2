# from classifier import app
from flask import render_template

from flask import Flask
app = Flask(__name__) # __name__という名称を付けてFlaskのインスタンス化
    # __name__はpythonがもともと確保している変数（予約語）であり、
    # 特別開発者が値を代入するコードを書かなくても最初からファイルの名称が代入されています。
    # 今回の場合は__name__には"app"(実行ファイル名)が代入されます。


# @~でルート設定して、その直下の関数が実行される = 2つで1セット
@app.route("/") # 「引数として渡しているパスに対してアクセスが来たら、この直下の関数を実行する」
def index():
    return render_template(
        'index_main.html',
    )


if __name__ == "__main__":

    # app.run() # Flaskアプリケーションを起動させています。
    #     # 起動後は先ほど説明した@app.route('/’)が効力を持つようになります。

    app.debug = False # True
    app.run(host='0.0.0.0', port=80) # 888)