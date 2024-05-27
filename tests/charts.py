from googleapiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials
from oauth2client.client import GoogleCredentials
from collections import defaultdict
import psycopg2
import sys
import json
import time
import os
import subprocess
import numpy as np
import pandas as pd
import plotly.express as px

d2l1 = []
d2l2 = []
d2l3 = []

file_path = 'datapoints2.txt'  # Replace 'your_file.txt' with your file path
with open(file_path, 'r') as file:
    for i in range(0, 47):
        file_content = file.readline()
        print(file_content)
        flats = file_content.split(' ')
        val1 = float(flats[0])
        print(val1)
        val2 = float(flats[1])
        val3 = float(flats[2]) 
        val4 = 0
        if (val1 < 1000):
            val4 = float(flats[3])
        d2l1.append(val1)
        d2l2.append(val2*1000)
        d2l3.append("SIMD-Smith-Waterman")
        d2l1.append(val1)
        d2l2.append(val3*1000)
        d2l3.append("NUPACK")
        if (val1 < 1000):
            d2l1.append(val1)
            d2l2.append(val4*1000)
            d2l3.append("RNAplex")
            

    d1table = {'# of characters': d2l1, 'time (ms)': d2l2, 'version': d2l3}
    d1tablef = pd.DataFrame(data=d1table)  
    print(d1tablef)
    fig = px.line(d1tablef, x = '# of characters', y = 'time (ms)', color='version', markers=True)
    fig.update_layout(font=dict(size=24))
    fig.show()



    
    
    
    

