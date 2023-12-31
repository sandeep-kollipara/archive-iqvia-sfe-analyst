# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:56:43 2020

@author: ksandeep
"""

import pandas
import openpyxl
import numpy

df = pandas.DataFrame({'Name': ['E','F','G','H'],'Age': [100,70,40,60]})

writer.book = openpyxl.load_workbook(r'C:\Users\ksandeep\demo.xlsx')

writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)

reader = pandas.read_excel(r'C:\Users\ksandeep\demo.xlsx', sheet_name='Sheet1')

df.to_excel(writer, sheet_name='Sheet1', index=False,header=False,startrow=len(reader)+1)

writer.save()