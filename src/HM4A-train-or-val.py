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

"hands-on/src/resnet/my_src/から移動"

"Target Object Name"
obj1 = "apple"
obj2 = "orange"

obj3 = "banana"
obj4 = "pine"

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
app = Flask(__name__, static_folder='upload_data') # デフォルトはstatic

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
        
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        filename = secure_filename(file.filename)  # ファイル名を安全なものに
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        
        # train_model = request.form['mode']
        r = request.form.get('mode')
        if r == "True":
            train_model = True
        else:
            train_model = False
        print("*** Current Mode : {} ***" .format(train_model))

        "*** 前処理 ***"
        imgs = Data_Preprocessing(filepath)
        # これでようやくNNに入力できるデータ形式になる
        "*** モデルのインスタンス化 ***"
        model = CNN(4)
        "*** モデルのロード ***"
        # train_model = True # 既存のデータで再度学習させてから推測するかどうか
        if train_model: # 再学習させたモデルをロード
            model = Retraind_and_Load_ReTrained_Model(model)
        else: # 学習済みモデルをロード
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


def Retraind_and_Load_ReTrained_Model(model):
    print("***** training *****")
    dir_path = "train_data"
    apple_files, orange_files, banana_files, pine_files, transform = pre_set(dir_path)
    apple_dataset = CatDogDataset(apple_files, dir_path, transform=transform)
    orange_dataset = CatDogDataset(orange_files, dir_path, transform=transform)
    banana_dataset = CatDogDataset(banana_files, dir_path, transform=transform)
    pine_dataset = CatDogDataset(pine_files, dir_path, transform=transform)
    # 二つのデータセットを結合して一つのデータセットにする
    dataset = ConcatDataset([apple_dataset, orange_dataset, banana_dataset, pine_dataset])
    # DataLoader作成
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last = True)
    train(data_loader, model)
    # パラメータの保存
    params = model.state_dict()
    torch.save(params, "re-train-model.param")

    return model

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


def pre_set(data_path):
    # data_path = "./train"
    data_path = data_path # "./my_data"

    
    file_list = os.listdir(data_path)
    
    apple_files = [file_name for file_name in file_list if obj1 in file_name]
    orange_files = [file_name for file_name in file_list if obj2 in file_name]

    banana_files = [file_name for file_name in file_list if obj3 in file_name]
    pine_files = [file_name for file_name in file_list if obj4 in file_name]

    transform = transforms.Compose([
        transforms.Resize((64, 64)), # 256, 256)), # 入力画像のサイズがバラバラなので、すべて64*64にリサイズする
        # transforms.ToTensor(),
        
        # # データ拡張のための前処理
        # transforms.RandomHorizontalFlip(), # ランダムに左右を入れ変える
        # transforms.ColorJitter(), # ランダムに画像の色値を変える
        # transforms.RandomRotation(10), # ランダムに画像の回転を行う(今回は10度)

        # EC2用
        # transforms.Resize((64, 64)), # 入力画像のサイズがバラバラなので、すべて64*64にリサイズする
        # # transforms.CenterCrop(224),  
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(), # ランダムに画像の色値を変える
        # transforms.RandomRotation(10), # ランダムに画像の回転を行う(今回は10度)
        # transforms.ToTensor(), 
        # # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.RandomAffine([-15, 15], scale=(0.8, 1.2)), # 回転(-15度~15度)とリサイズ(0.8倍~1.2倍)
        transforms.RandomHorizontalFlip(p=0.5),  # 左右反転
        transforms.ColorJitter(), # ランダムに画像の色値を変える
        transforms.ToTensor(), # EC2では最後 (訂正:領域消去や正規化の前) だが、社用PCでは最初にやらないとエラー

        transforms.RandomErasing(p=0.5), # 確率0.5でランダムに領域を消去
        transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),  # 平均値を0、標準偏差を1に [3チャンネル(RGB)なので3つある]
    ])

    
    # "Add"
    # val_transform = transforms.Compose([
    #     transforms.ToTensor(), # dataをtensor型に変換, channelを先頭にする
    #     transforms.Normalize((0.5,), (0.5,)) # channelごとに画像の平均値が0.5になるようにする
    # ])
    # train_transform = transforms.Compose([
    #     # データ拡張のための前処理
    #     transforms.RandomHorizontalFlip(), # ランダムに左右を入れける
    #     transforms.ColorJitter(), # ランダムに画像の色値を変える
    #     transforms.RandomRotation(10), # ランダムに画像の回転を行う(今回は10度)
    #     transforms.ToTensor(), # dataをtensor型に変換, channelを先頭にする
    #     transforms.Normalize((0.5,), (0.5,)) # channelごとに画像の平均値が0.5になるようにする
    # ])
    
    # train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    # validation_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=val_transform)
    # train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
    # "Add"

    return apple_files, orange_files, banana_files, pine_files, transform

def val_set():

    transform = transforms.Compose([
        transforms.Resize((64, 64)), # 256, 256)), # 入力画像のサイズがバラバラなので、すべて256*256にリサイズする
        transforms.ToTensor(),
        transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),  # 平均値を0、標準偏差を1に [3チャンネル(RGB)なので3つある]
    ])

    return transform

class CatDogDataset(Dataset):
    def __init__(self, file_list, dir, transform=None):
        # super().__init__()
        self.file_list = file_list
        self.dir = dir
        self.transform = transform

        # 正解ラベルの設定
        if obj1 in self.file_list[0]:
            self.label = 0
        elif obj2 in self.file_list[0]:
            self.label = 1
        elif obj3 in self.file_list[0]:
            self.label = 2
        elif obj4 in self.file_list[0]:
            self.label = 3
        else:
            self.label = -1
            print("***** Error *****")
    
    # 特殊メソッド
    def __len__(self):
        return len(self.file_list) # 画像の枚数を返す
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.dir, self.file_list[idx])
        img = Image.open(file_path)
        if self.transform is not None: # 前処理がある場合は前処理をする
            img = self.transform(img)
        return img, self.label

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # size:32*32
            # size:28*28

            # 通常ver.
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
            # "(A-1)"
            # # **1layer**
            # nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1), # in:256, out:128
            # nn.MaxPool2d(kernel_size=2),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1), # in:256, out:128
            # nn.ReLU(inplace=True),
            # # nn.Softmax()

            # "(B)"
            # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), # in:256, out:128
            # nn.ReLU(inplace=True),


            # nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1), # in:256, out:128
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
            # nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=1), # in:256, out:128
            # nn.ReLU(inplace=True),

            # **軽量化ver.**
            # # **1layer**
            # nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2), # in:3, out:64
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2), # sizeは1/2になる # size:16*16
            # # **1layer**
            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # in:256, out:128
            # nn.ReLU(inplace=True),
        )
        # ここまでが特徴抽出

        # 全結合
        # self.classifier = nn.Linear(in_features=4*4*64, out_features=num_classes) # 128, out_features=num_classes) # size(h) * size(w) * out_channels
        self.classifier = nn.Linear(in_features=8*8*128, out_features=num_classes) # size(h) * size(w) * out_channels
        # self.classifier = nn.Linear(in_features=7*7*64, out_features=num_classes) # size(h) * size(w) * out_channels

        # **軽量化ver.**
        # self.classifier = nn.Linear(in_features=16*16*128, out_features=num_classes) # size(h) * size(w) * out_channels


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

def train(train_dataloader, model):

    num_epochs = 50 # 15
    losses = [] # 損失
    accs = [] # 精度

    for epoch in range(num_epochs):

        # model.train() # train mode # 今回の記述方法(trainとvalのforを分ける)だとあまり結果に影響なさそう

        "(1)"
        # runnin_loss = 0.0
        # running_acc = 0.0
        "-----"
        "(2)"
        "----- E資格 -----"
        total_correct = 0
        total_data_len = 0
        total_loss = 0
        "-----------------"

        # 比較用
        # imgs_save

        # 8回ループ
        count = 0
        for imgs, labels in train_dataloader: # 1loopで32(バッチサイズ)個のデータを処理する
            # 画像データを1次元に変換
            # imgs = imgs.view(num_batches, -1) # 画像の順番になっているので、バッチを先頭に持ってくる. 後ろは任意.
            
            
            "これだとimgsと参照しているアドレスが同じで書き換えられてしまうと思ったが大丈夫だった"
            imgs_save = imgs
            
            "***** 入力がカラー画像の時!!!!! 重要 *****"
            # <こっちはグレースケール用> imgs = imgs.reshape(-1, 28*28*1)  # 画像データを1次元に変換 グレースケールではなく、カラーなので、3チャネル!!!!!
            
            "CNNを使うときはモデルの中ですでに一次元に変換しているのでコメントアウト -> MLPもモデルの中に記述"
            # imgs = imgs.reshape(-1, 28*28*3)  # 画像データを1次元に変換 グレースケールではなく、カラーなので、3チャネル!!!!!
            "imgs = [32(batch), 28*28(size) *28(channel)]"

            "***** 入力がカラー画像の時!!!!! 重要 *****"
            
            # imgs = imgs.to(device)
            # labels = labels.to(device)
            
            # optimizer.zero_grad()
            model.optimizer.zero_grad()

            output = model(imgs) # 順伝播
            # loss = criterion(output, labels)

            
            "表示用に追加"
            # print(imgs, len(imgs)) # 96
            # print(labels, len(labels)) # 32
            # print("out:{}, labels:{}".format(output, labels)) # 100, 32
            # red = torch.argmax(output, dim=1)
            # print("red:{}".format(red)) # 100


            loss = model.criterion(output, labels)
            "(1)"
            # runnin_loss += loss.item()
            "-----"
            "(2)"
            total_loss += loss.item()
            "-----"

            "このやり方ならone-hotベクトルにしなくてもいいかも"
            pred = torch.argmax(output, dim=1) # dimが0だとバッチ方向(縦). 1だと分類方向(横:0~9の分類) # イメージ:[batch, 予測]
            # 1バッチ分 = 32 個のデータ分を 0, 1 判定
            # print("  pred:{}".format(pred))
            
            "(1)" # 簡単な方法
            # running_acc += torch.mean(pred.eq(labels).float()) # 正解と一致したものの割合=正解率を計算
            "-----"
            "(2)" # より原始的な方法
            batch_size = len(labels)  # バッチサイズの確認
            for i in range(batch_size):  # データ一つずつループ,ミニバッチの中身出しきるまで
                total_data_len += 1  # 全データ数を集計
                if pred[i] == labels[i]:
                    total_correct += 1 # 正解のデータ数を集計
            "-----"

            loss.backward() # 逆伝播で勾配を計算
            # optimizer.step()
            model.optimizer.step() # 勾配をもとにパラメータを最適化


            # print("   total_data:{}".format(count)) # total_data_len)) # loop num : 0, 1, 2 = 3
            # print("   *** pred   : {} ***".format(pred))
            # print("   *** labels : {} ***".format(labels))
            count += 1
        
        "(1)"
        # runnin_loss /= len(train_dataloader)
        # running_acc /= len(train_dataloader)
        # losses.append(runnin_loss)
        # accs.append(running_acc)
        # print("epoch: {}, loss: {}, acc: {}".format(epoch, runnin_loss, running_acc))
        "-----"
        "(2)"
        accuracy = total_correct/total_data_len*100  # 予測精度の算出
        loss = total_loss/total_data_len  # 損失の平均の算出
        # print(f'正答率: {accuracy}, 損失: {loss}')
        print("epoch: {}, loss: {}, acc: {}".format(epoch, loss, accuracy))
        "-----"
        losses.append(loss)
        accs.append(accuracy)

    # train_iter = iter(train_dataloader)
    # imgs, labels = train_iter.__next__()
    # print(labels)

    # # パラメータの保存
    # params = model.state_dict()
    # torch.save(params, "model.param")
    

    # 最後の予測結果32個と正解ラベル32個を比較
    # だけど今は72/32=2...8なので、最後は8枚のみ出力される
    print("*************************************************************************************************************")
    print("予測結果  : {}".format(pred))
    
    # data_iter = iter(train_dataloader)
    # imgs, labels = data_iter.__next__() # 1バッチ分表示(size=32)
    print("正解ラベル: {}".format(labels))
    print("*************************************************************************************************************")
    grid_imgs = torchvision.utils.make_grid(imgs_save[:32]) # 24]) # 32枚表示
    grid_imgs_arr = grid_imgs.numpy()
    plt.figure(figsize=(16, 24))
    plt.imshow(np.transpose(grid_imgs_arr, (1, 2, 0)))
    # plt.show()
    plt.savefig('result-figure.png')
    plt.close()
    print("アドレス: {}, {}".format(id(imgs), id(imgs_save)))
    print("type: {}".format(type(imgs_save)))

    epochs = range(len(accs))
    # plt.style.use("ggplot")
    # plt.plot(epochs, losses, label="train loss")
    # # plt.figure()
    # plt.plot(epochs, accs, label="accurucy")
    # plt.legend()
    # # plt.show()
    # plt.savefig('result-data.png')
    # plt.close()

    fig = plt.figure(figsize=(15, 8))
    # add_subplot()でグラフを描画する領域を追加する．引数は行，列，場所
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    fig.suptitle("Train Result - loss & acc -")
    plt.style.use("ggplot")
    ax1.plot(epochs, losses, label="train loss", color = "orange")
    ax2.plot(epochs, accs, label="accurucy", color = "green")
    plt.legend()
    plt.savefig('result.png')

def validation(val_dataloader, model):
    # # パラメータの読み込み
    # param_load = torch.load("model.param")
    # model.load_state_dict(param_load)

    # model.eval() # eval mode # 今回の記述方法(trainとvalのforを分ける)だとあまり結果に影響なさそう

    total_correct = 0
    total_data_len = 0

    for imgs, labels in val_dataloader: # train_dataloader:

        # 画像データを一次元に変換
        "There are two ways to change data"
        "How to (1)"# E資格講座

        # imgs = imgs.reshape(-1, 28*28*1)  # 画像データを1次元に変換
        "***** 入力がカラー画像の時!!!!! 重要 *****"
        # imgs = imgs.reshape(-1, 28*28*1)  # 画像データを1次元に変換 グレースケールではなく、カラーなので、3チャネル!!!!!
        
        "CNNを使うときはモデルの中ですでに一次元に変換しているのでコメントアウト -> MLPもモデルの中に記述"
        # imgs = imgs.reshape(-1, 28*28*3)  # 画像データを1次元に変換 グレースケールではなく、カラーなので、3チャネル!!!!!
        "imgs = [32(batch), 28*28(size) *28(channel)]"

        "***** 入力がカラー画像の時!!!!! 重要 *****"
        
        "How to (2)"# 今回
        # imgs = imgs.view(num_batches, -1) # 画像の順番になっているので、バッチを先頭に持ってくる. 後ろは任意.
        "結局どちらも画像データ(28*28など)を後ろに持ってきている. -1は型に合うように揃えてくれている."
        "---------------------------------"

        # GPUメモリに送信
        # imgs = imgs.to(device)
        # labels = labels.to(device)
        
        output = model(imgs)
        
        # テストデータで検証
        "There are two ways to evaluate"
        "How to (1)"# ミニバッチごとの集計(E資格講座)
        # _, pred_labels = torch.max(output, axis=1)  # outputsから必要な情報(予測したラベル)のみを取り出す。
        # batch_size = len(labels)  # バッチサイズの確認
        # for i in range(batch_size):  # データ一つずつループ,ミニバッチの中身出しきるまで
        #     total_data_len += 1  # 全データ数を集計
        #     if pred_labels[i] == labels[i]:
        #         total_correct += 1 # 正解のデータ数を集計
        "How to (2)"# predの部分がE資格講座と異なる
        pred = torch.argmax(output, dim=1) # dimが0だとバッチ方向(縦). 1だと分類方向(横:0~9の分類) # イメージ:[batch, 予測]
        batch_size = len(labels)  # バッチサイズの確認
        for i in range(batch_size):  # データ一つずつループ,ミニバッチの中身出しきるまで
            total_data_len += 1  # 全データ数を集計
            if pred[i] == labels[i]:
                total_correct += 1 # 正解のデータ数を集計
        "------------------------------"
    
    # 今回のエポックの正答率と損失を求める
    accuracy = total_correct/total_data_len*100  # 予測精度の算出

    print("*************************************************************************************************************")
    print("予測結果  : {}".format(pred))
    
    # data_iter = iter(train_dataloader)
    # imgs, labels = data_iter.__next__() # 1バッチ分表示(size=32)
    print("正解ラベル: {}".format(labels))
    print("*************************************************************************************************************")
    
    print("正解率: {}".format(accuracy))

if __name__ == "__main__":
    # app.run(debug=True)
    app.debug = True
    app.run(host='0.0.0.0', port=80) # 888) # 0)