import os
import datetime
import pandas

# ENSURE THE DRIVES ARE MAPPED BEFORE RUNNING THE SCRIPT
# ENSURE ALL ADDRESSES USE WINDOWS FORMAT (\) NOT LINUX FORMAT (/)

fileaddress=input("Enter the address of the the 'Expected filelist' of Input tracker:");    # X:\Data\Interne_studies\Sandeep Kollipara\InputTrackerManual\Input_File_Tracker Curr_2019(expected).xlsx
sheetname=input("Enter the name of the Excel sheet:");    # General
fileset=pandas.read_excel(os.path.normpath(fileaddress), sheet_name=sheetname);
fileset['Previously Modified']=fileset['Date Modified'].copy();
i=0;
while i < len(fileset):
    filename=fileset.iloc[i][1];
    if os.path.exists(filename):
        fileset['Date Modified'][i]=str(datetime.datetime.fromtimestamp(os.path.getmtime(filename)).strftime('%Y-%m-%d %H:%M:%S'));
        fileset['Status'][i]="Fine";
    else:
        fileset['Date Modified'][i]=str(datetime.datetime.fromtimestamp(0).strftime('%Y-%m-%d %H:%M:%S'));
        fileset['Status'][i]="Does not exist/No access";
    i+=1;
fileset.to_excel(fileaddress, sheet_name=str(datetime.datetime.today().year)+str(datetime.datetime.today().month), index=False);
print("File list updated - Execution ended.")