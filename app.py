import os
from posixpath import dirname
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

PORT = os.environ.get('PORT', 3000)

@app.route('/')
def index():
    return render_template('hacknroll24.html')


@app.route('/save-video' , methods = ['POST', 'GET'])
def change_home_wallpaper():
    return redirect("/")


if __name__ == '__main__':
    app.run(port=PORT, debug=True)
    