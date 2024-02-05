from flask import Flask, request, redirect, url_for, render_template # , Markup
from markupsafe import Markup
from werkzeug.utils import secure_filename

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import os
import shutil
from PIL import Image
import numpy as np

UPLOAD_FOLDER = "./static/images/"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

labels = ["飛行機", "自動車", "鳥", "猫", "鹿", "犬", "カエル", "馬", "船", "トラック"]
n_class = len(labels)
img_size = 32
n_result = 3  # 上位3つの結果を表示

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS # 許容する拡張子内に含まれているならTrueを返す


@app.route("/", methods=["GET", "POST"]) #  直下のフォルダに来た場合は直下のindex関数を実行
def index():
    return render_template("index.html") # htmlを表示

"***** 重要 *****"
# このURLを指定してアクセスした場合、画像をアップロードすると以下の処理が行われて予測が出力される.
@app.route("/result", methods=["GET", "POST"]) # resultというURLに来た場合はresult関数を実行
def result():
    if request.method == "POST": # 画像をweb上に投稿(アップロード)
        # ファイルの存在と形式を確認
        if "file" not in request.files:
            print("File doesn't exist!")
            return redirect(url_for("index")) # Topに戻る
        file = request.files["file"]
        if not allowed_file(file.filename):
            print(file.filename + ": File not allowed!")
            return redirect(url_for("index")) # Topに戻る

        # ファイルの保存
        # if os.path.isdir(UPLOAD_FOLDER):
        #     shutil.rmtree(UPLOAD_FOLDER)
        # os.mkdir(UPLOAD_FOLDER)
        
        # add
        if not os.path.isdir(UPLOAD_FOLDER):
            os.mkdir(UPLOAD_FOLDER)
        filename = secure_filename(file.filename)  # ファイル名を安全なものに
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # 画像の読み込み
        image = Image.open(filepath)
        image = image.convert("RGB") # アップロードされた画像が「モノクロや透明値αが含まれるかもしれない」からRGBに変換して入力画像を統一
        image = image.resize((img_size, img_size)) # 入力画像を32*32にする

        normalize = transforms.Normalize(
            (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))  # 平均値を0、標準偏差を1に
        to_tensor = transforms.ToTensor()
        transform = transforms.Compose([to_tensor, normalize])

        x = transform(image)
        x = x.reshape(1, 3, img_size, img_size) # バッチサイズ, チャンネル数, 高さ, 幅

        # これでようやくNNに入力できるデータ形式になる

        
        
        
        
        
        # 予測
        net = Net()
        net.load_state_dict(torch.load(
            "model_cnn.pth", map_location=torch.device("cpu")))
        net.eval()  # 評価モード

        y = net(x)
        y = F.softmax(y, dim=1)[0] # 10個の出力が確率になる
        sorted_idx = torch.argsort(-y)  # 降順でソート # 大きい順で並べてindexをargsortで取得
        result = ""
        # 今回は結果を3つ表示
        for i in range(n_result):
            idx = sorted_idx[i].item() # 大きい順にソートしているので、最も大きい値が入る
            ratio = y[idx].item()
            label = labels[idx]
            result += "<p>" + str(round(ratio*100, 1)) + \
                "%の確率で" + label + "です。</p>"
        return render_template("result.html", result=Markup(result), filepath=filepath) # result.htmlにこの結果を表示
    else:
        return redirect(url_for("index")) # POSTがない場合はトップに戻る


if __name__ == "__main__":
    # app.run(debug=True)
    app.debug = True
    app.run(host='0.0.0.0', port=80) # 888) # 0)
