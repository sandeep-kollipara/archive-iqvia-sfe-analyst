import os
import pandas

fileaddress1=input("Enter the address of the first file:"); # \\rtdsfps01\userdata\customer support\lvs\SMS\2019\20xxMxx LVS SetSpecificatie.xlsx
sheetname1=input("Enter the name of the sheet in first file:"); # SETSPECIFICATIE
fileaddress2=input("Enter the address of the second file:");
sheetname2=input("Enter the name of the sheet in second file:");
datafile1=pandas.read_excel(os.path.normpath(fileaddress1), sheet_name=sheetname1);
datafile2=pandas.read_excel(os.path.normpath(fileaddress2), sheet_name=sheetname2);
print("Data files imported successfully.");
iframe1=datafile1.iloc[:,1:12].drop(columns=['aantal']).copy();
iframe2=datafile2.iloc[:,1:12].drop(columns=['aantal']).copy();
iframe11=iframe1.loc[iframe1['opmerkingen'].str.contains('Analyst CE:', na=False)];
iframe22=iframe2.loc[iframe2['opmerkingen'].str.contains('Analyst CE:', na=False)];
iframe111=iframe11.sort_values(by=list(iframe11.columns), ascending=True).reset_index(drop=True);
iframe222=iframe22.sort_values(by=list(iframe22.columns), ascending=True).reset_index(drop=True);
if iframe111.equals(iframe222):
    print("Both data files have equal data");
else:
    iframe3=pandas.concat([iframe111,iframe222]);
    iframe3=iframe3.drop_duplicates(keep=False);
    iframe3.to_excel("Unequal Rows.xlsx", sheet_name='ExcelEquals');
    print("Data files are not equal. Check 'Unequal Rows.xlsx' for unequal rows");