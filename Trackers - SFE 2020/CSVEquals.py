import os
import pandas

fileaddress1=input("Enter the address of the first file:"); # \\rtdsfps01\userdata\customer support\lvs\SMS\2019\20xxMxx LVS SetSpecificatie.xlsx
sheetname1=input("Enter the name of the sheet in first file:"); # SETSPECIFICATIE
fileaddress2=input("Enter the address of the second file:");
sheetname2=input("Enter the name of the sheet in second file:");
datafile1=pandas.read_csv(os.path.normpath(fileaddress1));
datafile2=pandas.read_csv(os.path.normpath(fileaddress2));
print("Data files imported successfully.");
if datafile1.equals(datafile2):
    print("Both data files have equal data");
else:
    iframe3=pandas.concat([iframe111,iframe222]);
    iframe3=iframe3.drop_duplicates(keep=False);
    iframe3.to_excel("Unequal Rows.xlsx", sheet_name='ExcelEquals');
    print("Data files are not equal. Check 'Unequal Rows.xlsx' for unequal rows");