import os
import time
import datetime
import pandas
import fnmatch

# ENSURE THE DRIVES ARE MAPPED BEFORE RUNNING THE SCRIPT
# ENSURE ALL ADDRESSES USE WINDOWS FORMAT (\) NOT LINUX FORMAT (/)

fileaddress=input("Enter the address of the the latest version of Input tracker:");    # X:\Data\Interne_studies\Documents\00 Project Trackers\00 Delivery Tracker\Input_File_Tracker Mmm_YYYY.xlsx
sheetname=input("Enter the name of the Excel sheet:");    # General
#currmonth=int(input("Enter the current month (in number):"));    # 1-12
#if not 0 < currmonth < 13:
#    print("Entered month value not supported. Program ending...");
#    exit();
datafile=pandas.read_excel(os.path.normpath(fileaddress), sheet_name=sheetname);
filelist=datafile.iloc[6:,1].to_frame();
print("Importing the wildcard filelist from <X:\Data\Interne_studies\Sandeep Kollipara>...");
wildcardfilelist=pandas.read_excel(os.path.normpath("X:\Data\Interne_studies\Sandeep Kollipara\InputTrackerWildCardFileList.xlsx"), sheet_name="Sheet");
if len(filelist) != len(wildcardfilelist):
    print("Numer of filenames mismatched. Please update the filelists. Program ending...");
    exit();
else:
    i=0;
    while i < len(filelist):
        wildcard=wildcardfilelist['Wild_Card_File_List'][i];
        wildcardmod=wildcard.replace("$","?");
        filename=filelist.iloc[i][0];
        if not fnmatch.fnmatch(filename, wildcardmod):
            print("Filenames out of order or improperly matched");
        i+=1;
print("Both the file lists are in order.");
deadline=datetime.datetime(datetime.datetime.now().year,datetime.datetime.now().month,1,00,00);
i=0;
register_template=pandas.DataFrame(columns=['File_Address','Updated_File_Address','Status'], data=[['-','-','-']]);
register=register_template.iloc[:0,:];
while i < len(filelist):
    filename=filelist.iloc[i][0];
    wildcard=wildcardfilelist['Wild_Card_File_List'][i];
    cardsplit=wildcard.split("$");
    cardsplit=list(set(cardsplit));
    if "" in cardsplit:
        cardsplit.remove("");
    j=0;
    for strand in cardsplit:
        if "\\" in strand:
            j+=1;
    if j > 2:
        print("Only 2 variations in address are supported. Program ending due to > 2 variations...");
        exit();
    elif j == 2:
        # both folder and file updated
        print("Cool^2!");
    elif j == 1:
        # only file updated
        cardsplit2=wildcard.rsplit("\\",1);
        if "$" in cardsplit2[1]:
            # namechanging file
            similarfilenames=[x for x in os.listdir(cardsplit2[0]) if fnmatch.fnmatch(x, cardsplit2[1])]
        else:
            # constant filename
            epochmodtime=os.path.getmtime(filename);
            if datetime.datetime.utcfromtimestamp(epochmodtime) > deadline:
                status='Updated';
            else:
                status='Outdated';
            updatedfilename=filename;
            print("Contant filename status determined.")
    else:
        print("File path is abnormal, please check file addresses. Program ending...");
        exit();
    record=register_template.copy();
    record['File_Address']=filename;
    record['Updated_File_Address']=updatedfilename;
    record['Status']=status
    register=register.append(record);
    i+=1;