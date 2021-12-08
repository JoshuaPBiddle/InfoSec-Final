from flask import Blueprint, render_template, request, redirect
import os
views = Blueprint('views', __name__)

@views.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
      if request.files:
         csv = request.files["csv"]
         csv.save(os.path.join("uploads", csv.filename))
         return redirect(request.url)
    return render_template("home.html")