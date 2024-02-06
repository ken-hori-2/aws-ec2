from flask import Flask, render_template, url_for, request
app = Flask(__name__, template_folder='../templates')

@app.route('/',methods=['GET'])
def index():
   values = {"name": "カマキリ", "age" :100}
   return render_template('index_test.html', \
       values=values,\
       title = "Flask入門",\
       message = 'Flask入門へようこそ！',)

@app.route('/', methods=['POST'])
def form():
   values = {"name": "カマキリ", "age" :100}
   name_kamakiri = request.form['kamakiri']
   ck = request.form.get('check')
   r = request.form.get('radio')
   list_datas = request.form.getlist('list_data')

   
   return render_template('index_test.html', \
       values=values,\
       title = "Flask入門",\
       message = '%sさん、こんにちは！ Flask入門へようこそ！'% name_kamakiri,\
       data = [ck, r, list_datas])

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=80)