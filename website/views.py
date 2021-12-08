from flask import Blueprint, render_template, request, redirect, current_app
import os
views = Blueprint('views', __name__)



@views.route("/", methods=["GET", "POST"])
def home():
    current_app.config["FILE_UPLOADS"] = "/c/Users/Joshua/Desktop/WebApp/website/static/uploads"
    if request.method == "POST":
      if request.files:
         csv = request.files["csv"]
         csv.save(os.path.join(current_app.config["FILE_UPLOADS"], csv.filename))
         return redirect(request.url)
    return render_template("home.html")