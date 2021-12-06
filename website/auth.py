from flask import Blueprint, render_template

auth = Blueprint('auth', __name__)

@auth.route('/chart')
def chart():
   return render_template("chart.html")

@auth.route('/report')
def report():
   return render_template("report.html")