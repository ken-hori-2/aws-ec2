import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import os
import time
from PIL import Image
from flask import Flask, request, redirect, url_for, render_template # , Markup
from markupsafe import Markup
from werkzeug.utils import secure_filename
import torch.nn.functional as F





# horiuchi-model-for-App.pyの整理ver.





# 自分で集めたデータを前処理してモデルに入力
# my_data内の画像から4種類を分類するモデル

# UPLOAD_FOLDER = "./static/images/" # デフォルトはstatic
UPLOAD_FOLDER = "./upload_data/images/" # 判別するためにアップロードした画像を保存しておくフォルダ
# > いずれこれも学習データに統合する


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

labels = ["りんご", "みかん", "バナナ", "パイナップル"]
n_class = len(labels)
img_size = 64 # 32
n_result = 3  # 上位3つの結果を表示
# app = Flask(__name__)
# 相対パスで指定
app = Flask(__name__, static_folder='upload_data', template_folder='../my_templates') # デフォルトはstatic

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

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
        
        # add
        # if not os.path.isdir(UPLOAD_FOLDER):
        if not os.path.exists(UPLOAD_FOLDER):
            # os.mkdir(UPLOAD_FOLDER)
            os.makedirs(UPLOAD_FOLDER)
        filename = secure_filename(file.filename)  # ファイル名を安全なものに
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        "*** 前処理 ***"
        imgs = Data_Preprocessing(filepath)
        # これでようやくNNに入力できるデータ形式になる
        "*** モデルのインスタンス化 ***"
        model = CNN(4)
        "*** モデルのロード ***"
        # # train_model = True # 既存のデータで再度学習させてから推測するかどうか
        # if train_model: # 再学習させたモデルをロード
        #     model = Retraind_and_Load_ReTrained_Model(model)
        # else: # 学習済みモデルをロード
        model = Load_Trained_Model(model)
        "*** 予測結果 ***"
        pred = Prediction_Result(imgs, model)
        result = Display_Three_Results(pred, n_result)

        
        return render_template("result.html", result=Markup(result), filepath=filepath) # result.htmlにこの結果を表示
    else:
        return redirect(url_for("index")) # POSTがない場合はトップに戻る

def Data_Preprocessing(filepath):
    # 画像の読み込み
    image = Image.open(filepath)
    image = image.convert("RGB") # アップロードされた画像が「モノクロや透明値αが含まれるかもしれない」からRGBに変換して入力画像を統一    
    "***** コメントアウト *****"
    # image = image.resize((img_size, img_size)) # 入力画像を32*32にする
    # > val_set()で処理する

    # normalize = transforms.Normalize(
    #     (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))  # 平均値を0、標準偏差を1に
    # to_tensor = transforms.ToTensor()
    # transform = transforms.Compose([to_tensor, normalize])
    # x = transform(image)
    # x = x.reshape(1, 3, img_size, img_size) # バッチサイズ, チャンネル数, 高さ, 幅
    # # x = x.reshape(x.size(0), -1) # データを1次元に変換 # 上と同じ処理
    "***** コメントアウト *****"

    "val用のtransformを作成"
    val_transform = val_set()
    imgs = val_transform(image)
    # for imgs, labels in val_dataloader: # 1loopで32(バッチサイズ)個のデータを処理する
    imgs = imgs.reshape(1, 3, img_size, img_size) # dataloaderの上記のような処理が必要（データの一次元化） # バッチサイズ, チャンネル数, 高さ, 幅

    return imgs

def Load_Trained_Model(model):
    print("***** validation *****")
    # パラメータの読み込み
    param_load = torch.load("model-for-ec2.param")
    model.load_state_dict(param_load)
    # validation(validation_loader, model)
    
    return model

def Prediction_Result(imgs, model):
    output = model(imgs)
    pred = F.softmax(output, dim=1)[0] # 10個の出力が確率になる
    # pred = torch.argmax(output, dim=1) # dimが0だとバッチ方向(縦). 1だと分類方向(横:0~9の分類) # イメージ:[batch, 予測]
    # ↑これだと、4つの分類から最大値のみを選んでしまうのでだめ

    return pred

def Display_Three_Results(pred, n_result):
    sorted_idx = torch.argsort(-pred)  # 降順でソート # 大きい順で並べてindexをargsortで取得
    result = ""
    # 今回は結果を3つ表示
    for i in range(n_result):
        idx = sorted_idx[i].item() # 大きい順にソートしているので、最も大きい値が入る
        ratio = pred[idx].item()
        label = labels[idx]
        result += "<p>" + str(round(ratio*100, 1)) + \
            "%の確率で" + label + "です。</p>"
    
    return result

def val_set():

    transform = transforms.Compose([
        transforms.Resize((64, 64)), # 256, 256)), # 入力画像のサイズがバラバラなので、すべて256*256にリサイズする
        transforms.ToTensor(),
        transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),  # 平均値を0、標準偏差を1に [3チャンネル(RGB)なので3つある]
    ])

    return transform

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # **1layer**
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2), # in:3, out:64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # sizeは1/2になる # size:16*16 # size:14*14
            # **1layer**
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # in:64, out:128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # sizeは1/2になる # size:8*8 # size:7*7
            # **1layer**
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1), # in:128, out:256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # sizeは1/2になる # size:4*4
            # "(A-0)"
            # **1layer**
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1), # in:256, out:128
            # nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1), # in:256, out:128
            # nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )
        # ここまでが特徴抽出

        # 全結合
        # self.classifier = nn.Linear(in_features=4*4*64, out_features=num_classes) # 128, out_features=num_classes) # size(h) * size(w) * out_channels
        self.classifier = nn.Linear(in_features=8*8*128, out_features=num_classes) # size(h) * size(w) * out_channels
        # self.classifier = nn.Linear(in_features=7*7*64, out_features=num_classes) # size(h) * size(w) * out_channels

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=5e-4) # 重み付けが大きくなりすぎないようにする
    
    def forward(self, x):
        
        x = self.features(x) # 4*4サイズ, 128チャネルの画像が出力
        "(1)"
        # x = x.view(x.size(0), -1) # 1次元のベクトルに変換 # size(0) = batch数 = 32
        # ** x = [batch, c * h * w] になる **
        "-----"
        "(2)"
        # x = x.view(num_batches, -1) # 画像の順番になっているので、バッチを先頭に持ってくる. 後ろは任意.
        # x = x.reshape(-1, 4*4*128)  # 画像データを1次元に変換
        # x = x.reshape(-1, 16*16*128)  # 軽量化ver.
        x = x.reshape(x.size(0), -1) # データを1次元に変換
        "-----"
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    # app.run(debug=True)
    app.debug = True
    app.run(host='0.0.0.0', port=80) # 888) # 0)