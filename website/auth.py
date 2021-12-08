from flask import Blueprint, render_template, request, redirect
import pandas as pd
import numpy as np
import csv
from flask import send_file

from pandas.core.arrays.string_ import StringArray
auth = Blueprint('auth', __name__)

@auth.route('/chart', methods=['GET', 'POST'])
def chart():
   return render_template("chart.html")

@auth.route('/report', methods=['GET', 'POST'])
def report():
   filename = "output.csv"
   myData = pd.read_csv(filename, header=0)
   data = myData.values
   return render_template("report.html", data = data)

@auth.route('/email',  methods=['GET', 'POST'])
def email():
   data = request.args.get('value', None)

   return render_template("email.html", data=data )

@auth.route('/download')
def downloadFile():
    filename = "output.csv"
    return send_file(filename, as_attachment=True)

