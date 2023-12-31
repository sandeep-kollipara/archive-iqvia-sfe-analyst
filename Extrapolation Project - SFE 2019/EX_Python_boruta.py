import os
import numpy
import pandas
import collections
import datetime
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier
from sklearn.feature_selection import RFE
from boruta import BorutaPy    #If not installed, carry out the following command in 'Anaconda Prompt'(inside square brackets): [conda install -c conda-forge boruta_py]
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE    #If not installed, carry out the following command in 'Anaconda Prompt'(inside square brackets): [conda install -c glemaitre imbalanced-learn]

#def UnivariateAnalysis(inputframe) DEPRECATED

def EliminateConstantFeatures(dataframe):
    featurelist=dataframe.columns;
    integertypes=['int16', 'int32', 'int64'];
    floattypes=['float16', 'float32', 'float64'];
    for feature in featurelist:
        if True in dataframe[feature].isnull().unique():
            continue;
        elif dataframe[feature].dtype in integertypes and len(dataframe[feature].unique())==1:
            dataframe.drop([feature], axis=1, inplace=True);
            print("Feature of int class: "+str(feature)+" dropped.");
        elif dataframe[feature].dtype in floattypes and dataframe[feature].std()==0:
            dataframe.drop([feature], axis=1, inplace=True);
            print("Feature of float class: "+str(feature)+" dropped.");
        elif dataframe[feature].dtype==object and len(dataframe[feature].unique())==1:
            dataframe.drop([feature], axis=1, inplace=True);
            print("Feature of object class: "+str(feature)+" dropped.");
    print("Num. of features reduced to "+str(len(featurelist))+\
          " from "+str(len(dataframe.columns))+" after Constant Feature Elimination.");
    return dataframe;

def EliminateFeaturesWithHighNulls(dataframe, percentage):
    featurelist=dataframe.columns;
    dflen=len(dataframe);
    for feature in featurelist:
        null_dict=collections.Counter(dataframe[feature].isnull());
        if null_dict[True] > percentage*dflen/100:
            dataframe.drop([feature], axis=1, inplace=True);
    print("Num. of features reduced to "+str(len(featurelist))+\
          " from "+str(len(dataframe.columns))+" after Features with High Nulls Elimination.");
    return dataframe;

def EliminateIdenticalFeatures(dataframe):#returns a different dataframe
    dataframe2=dataframe.copy();
    featurelist=dataframe.columns;
    newfeaturelist=dataframe.columns;
    for feature_1 in featurelist:
        for feature_2 in featurelist:
            if feature_1==feature_2:
                continue;
            else:
                if dataframe[feature_1].dtype != dataframe[feature_2].dtype:
                    continue;
                else:
                    temp_1=pandas.DataFrame(dataframe[feature_1].fillna(method='pad'));
                    temp_1=temp_1.rename(columns={feature_1:'A'});
                    temp_2=pandas.DataFrame(dataframe[feature_2].fillna(method='pad'));
                    temp_2=temp_2.rename(columns={feature_2:'A'});
                    if len(temp_1) != len(temp_2) or len(temp_1)==0 or len(temp_2)==0:
                        continue;
                    else:
                        if temp_1.equals(temp_2) and feature_1 in newfeaturelist and feature_2 in newfeaturelist:
                            dataframe2.drop([feature_2], axis=1, inplace=True);
                            print("Feature "+str(feature_1)+" and "+str(feature_2)+" are identical.");
                            newfeaturelist=newfeaturelist.drop(feature_2);
    dataframe=dataframe2;
    return dataframe2;

def EliminateFeaturesWithHighZeroes(dataframe, percentage):
    featurelist=dataframe.columns;
    dflen=len(dataframe);
    for feature in featurelist:
        if dataframe[feature].dtype != object:
            zero_dict=collections.Counter(dataframe[feature]);
            if zero_dict[0] > percentage*dflen/100:
                dataframe.drop([feature], axis=1, inplace=True);
                print("Feature "+str(feature)+" dropped.");
    print("Num. of features reduced to "+str(len(featurelist))+\
          " from "+str(len(dataframe.columns))+" after Features with High Zeroes Elimination.");
    return dataframe;

def EliminateFeaturesWithNearZeroVariance(dataframe):
    selector=VarianceThreshold();
    labelenc=LabelEncoder();
    dfcol=len(dataframe.columns);
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'];
    dataframe.copy();
    for feature in dataframe.columns:
        temp=dataframe[feature].copy();
        temp=temp.dropna(axis=0, how='any').reset_index();
        if temp[feature].dtype not in numerics:
            temp[feature]=labelenc.fit_transform(temp[feature].astype(str));
        selector.fit(temp);
        if feature not in temp.columns[selector.get_support(indices=True)]:
            dataframe.drop([feature], axis=1, inplace=True);
            print("Feature "+str(feature)+" dropped.");
    print("Num. of features reduced to "+str(len(dataframe.columns))+\
          " from "+str(dfcol)+\
          " after Features with Near Zero Variance Elimination.");
    return dataframe;

def EliminateFeaturesWithHighNonnumericUniqueValues(dataframe, percentage, absolute):
    featurelist=dataframe.columns;
    numerics=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'];
    nonnumfeaturelist=featurelist.drop(dataframe.select_dtypes(include=numerics).columns);
    dfcol=len(dataframe.columns);
    for feature in nonnumfeaturelist:
        if len(dataframe[feature].unique())/len(dataframe) > percentage/100:
            dataframe.drop([feature], axis=1, inplace=True);
            print("Feature "+str(feature)+" dropped.");
        elif len(dataframe[feature].unique()) > absolute:
            dataframe.drop([feature], axis=1, inplace=True);
            print("Feature "+str(feature)+" dropped.");
    print("Num. of features reduced to "+str(len(dataframe.columns))+\
          " from "+str(dfcol)+\
          " after Features with High Non-numeric Unique Values Elimination.");
    return dataframe;

def EliminateCorrelatedFeatures(dataframe, threshold):
    correlation_matrix=dataframe.corr().reset_index();
    dfcol=len(dataframe.columns);
    side=len(correlation_matrix);
    featurelist=correlation_matrix.columns;
    newfeaturelist=correlation_matrix.columns;
    featurelist=featurelist.drop('index');
    j=side;
    for feature in featurelist:
        i=side-j+1;
        while i < side:
            if correlation_matrix[feature][i] > threshold \
            and feature in newfeaturelist \
            and correlation_matrix['index'][i] in newfeaturelist:
                dataframe.drop([correlation_matrix['index'][i]], axis=1, inplace=True);
                print("Feature "+str(correlation_matrix['index'][i])+" dropped.");
                newfeaturelist=newfeaturelist.drop(correlation_matrix['index'][i]);
            i+=1;
        j-=1;
    print("Num. of features reduced to "+str(len(dataframe.columns))+\
          " from "+str(dfcol)+" after Correlated Features Elimination.");
    return dataframe;

def MissingValueTreatment(dataframe):
    featurelist=dataframe.columns;
    integertypes=['int16', 'int32', 'int64'];
    floattypes=['float16', 'float32', 'float64'];
    numerics=list(set(integertypes+floattypes));
    intfeaturelist=dataframe.select_dtypes(include=integertypes).columns;
    floatfeaturelist=dataframe.select_dtypes(include=floattypes).columns;
    nonnumfeaturelist=featurelist.drop(dataframe.select_dtypes(include=numerics).columns);
    for feature in intfeaturelist:
        dataframe[feature]=dataframe[feature].fillna(round(dataframe[feature].mean()));
    for feature in floatfeaturelist:
        dataframe[feature]=dataframe[feature].fillna(dataframe[feature].mean());
    for feature in nonnumfeaturelist:
        dataframe[feature]=dataframe[feature].fillna(dataframe[feature].mode().item());
    return dataframe;

def OutlierTreatment(dataframe):
    floattypes=['float16', 'float32', 'float64'];
    floatfeaturelist=dataframe.select_dtypes(include=floattypes).columns;
    for feature in floatfeaturelist:
        dataframe[feature].loc[dataframe[feature] < dataframe[feature].quantile(0.01)]=dataframe[feature].quantile(0.01);
        dataframe[feature].loc[dataframe[feature] > dataframe[feature].quantile(0.99)]=dataframe[feature].quantile(0.99);
    return dataframe;

#-----------------------------------main--------------------------------------

if __name__=='__main__':
    
    class ExitScript(Exception): pass;
    
    # Importing Raw data
    option=False;
    while not option:
        try:
            csvfilename=input("Enter the name of the csv file to be imported(w/ext.):");
            rawframe=pandas.read_csv(csvfilename, encoding='latin1');           #backup
            data=pandas.read_csv(csvfilename, encoding='latin1');               #working
            print("Data file successfully imported.");
            option=True;
        except AttributeError:
            print("There's no item with that code");
        except KeyError:
            print("Bad parameter name");
        except:
            print("Unknown error");
    
    # Univariate Analysis - DEPRECATED
    #univariate_report=UnivariateAnalysis(rawframe); - DEPRECATED
    #univariate_report.to_csv("EX_Univariate_Report.csv"); - DEPRECATED
    
    # Automatically Removing Axis params
    axisnum=1;
    while "Y"+str(axisnum) in data.columns: # Axis Sales
        data.drop(["Y"+str(axisnum)], axis=1, inplace=True);
        axisnum+=1;
    axisnum=1;
    while "L"+str(axisnum) in data.columns: # Axis Segments
        data.drop(["L"+str(axisnum)], axis=1, inplace=True);
        axisnum+=1;
    # Manually Removing Known Non-useful features
    data.drop(["ID","FLAG"], axis=1, inplace=True) # ID, Labels etc.
    data.drop(["var1","var2"], axis=1, inplace=True) # name variables
    data.drop(["var4","var5"], axis=1, inplace=True) # var6 covers this info
    data.drop(["var7","var8"], axis=1, inplace=True) # var9 covers this info
    data.drop(["var10"], axis=1, inplace=True) # var11 covers this info
    data.drop(["var15","var16"], axis=1, inplace=True) # var17 covers this info
    data.drop(["var20"], axis=1, inplace=True) # var19 covers this info
    data.drop(["var28"], axis=1, inplace=True) # var29 covers this info
    data.drop(["var30"], axis=1, inplace=True) # Wkp_Intl_Onekey_ID
    data.drop(["var32"], axis=1, inplace=True) # Parent_Wkp_Intl_Onekey_ID
    data.drop(["var34"], axis=1, inplace=True) # Parent2_Wkp_Intl_Onekey_ID
    data.drop(["var36"], axis=1, inplace=True) # Addr_Address
    data.drop(["var37"], axis=1, inplace=True) # Addr_PostCode
    data.drop(["var38"], axis=1, inplace=True) # Addr_City
    data.drop(["var39"], axis=1, inplace=True) # Wkp_Web_URL
    data.drop(["var40"], axis=1, inplace=True) # Phone
    data.drop(["var41"], axis=1, inplace=True) # Brick1_cod
    data.drop(["var42"], axis=1, inplace=True) # Brick2_cod
    data.drop(["var43"], axis=1, inplace=True) # Brick3_cod
    data.drop(["var45"], axis=1, inplace=True) # var46 covers this info
    data.drop(["var47"], axis=1, inplace=True) # 2nd_Parent_Wkp_Intl_Onekey_ID
    data.drop(["var49"], axis=1, inplace=True) # var50 covers this info
    data.drop(["var51"], axis=1, inplace=True) # var52 covers this info
    
    # Variable Transformations - this phase is executed earlier than R counterpart
    # Graduation Year - var22
    curryear=int(str(datetime.datetime.today())[:4]);# Graduation Year --> Years since graduated
    if 'var22' not in data.columns:
        print("Doctor Graduation information missing: Column var22.");
        #exit();
        raise ExitScript("************* PROGRAM END *************");
    else:
        if data['var22'].dtype not in [int, float]:
            data['var22']=data['var22'].astype(float);
        data['var22']=data.apply(lambda cell: curryear-cell['var22'], axis=1);
        print("'Years since graduated' updated for Doctor Graduation info successfully.");
    # Workplace Re-classification - var29 (Workplace type information --> Classify as Academic/NonAcademic)
    if 'var29' not in data.columns:
        print("Workplace type information missing: Column var29.");
        #exit();
        raise ExitScript("************* PROGRAM END *************");
    else:
        if data['var29'].dtype != object:
            data['var29']=data['var29'].astype(str);
        #ADAPT: ADD NEW WORKPLACES HERE IF PRESENT
        data['var29'].loc[data['var29']=="Academic Hospital"]="Academic";
        data['var29'].loc[data['var29']=="Barracks"]="NonAcademic";
        data['var29'].loc[data['var29']=="Coordinating Organisations"]="NonAcademic";
        data['var29'].loc[data['var29']=="Expertise Center"]="NonAcademic";
        data['var29'].loc[data['var29']=="General Hospital Center"]="NonAcademic";
        data['var29'].loc[data['var29']=="Health Center"]="NonAcademic";
        data['var29'].loc[data['var29']=="Hospital Board"]="NonAcademic";
        data['var29'].loc[data['var29']=="Medical Association"]="NonAcademic";
        data['var29'].loc[data['var29']=="Medical Center"]="NonAcademic";
        data['var29'].loc[data['var29']=="Medical Committee"]="NonAcademic";
        data['var29'].loc[data['var29']=="Other Institution"]="Academic";
        data['var29'].loc[data['var29']=="Practice"]="NonAcademic";
        data['var29'].loc[data['var29']=="Private Address"]="NonAcademic";
        data['var29'].loc[data['var29']=="Research Center"]="Academic";
        data['var29'].loc[data['var29']=="Company"]="NonAcademic";
        data['var29'].loc[data['var29']=="Gov. Mental Health Centre"]="NonAcademic";
        data['var29'].loc[data['var29']=="Hospital Division"]="NonAcademic";
        data['var29'].loc[data['var29']=="Municipal Health Services"]="NonAcademic";
        data['var29'].loc[data['var29']=="Nursing Home"]="NonAcademic";
        data['var29'].loc[data['var29']=="Education"]="Academic";
        data['var29'].loc[data['var29']=="Patients Association"]="NonAcademic";
        data['var29'].loc[data['var29']=="Care groups"]="NonAcademic";
        data['var29'].loc[data['var29']=="Company healthcare"]="NonAcademic";
        data['var29'].loc[data['var29']=="Public Administration"]="NonAcademic";
        data['var29'].loc[data['var29']=='Psychiatric Hospital']="NonAcademic";
        data['var29'].loc[data['var29']=='Home Care']="NonAcademic";
        data['var29'].loc[data['var29']=='Day Nursery']="NonAcademic";
        data['var29'].loc[data['var29']=='Home of Rest (Elderly People)']="NonAcademic";
        #data['var29'].loc[data['var29']=="<INSERT NEW ITEM HERE>"]="NonAcademic"; - DO NOT DELETE
    if len(list(collections.Counter(data['var29']))) != 2:
        print("Workplace list needs to be updated: Column var29.");
        #exit();
        raise ExitScript("************* PROGRAM END *************");
        #READ ME:
        #In case of error here, manually run this line of code below:
        #data['var29'].unique();
        #If the array contains anything other than 'Academic' and 'NonAcademic', then add
        #the same to the template in the transformation list above
        #After the changes, restart frest!
    else:
        print("Workplace parameter relabelled successfully.");
    
    # Eliminating Features
    try:
        os.mkdir('Variables Check');
    except:
        print("Address already exists.");
    pandas.DataFrame(list(data.columns), columns =['Var Names']).to_excel("Variables Check\\01.Variables_Post_Manual_Elimination.xlsx", sheet_name='Data', index=False);
    EliminateConstantFeatures(data); # improved over R code: additionally 'var6' also removed
    pandas.DataFrame(list(data.columns), columns =['Var Names']).to_excel("Variables Check\\02.Variables_Post_Constant_Feature_Elimination.xlsx", sheet_name='Data', index=False);
    EliminateFeaturesWithHighNulls(data, 60);
    pandas.DataFrame(list(data.columns), columns =['Var Names']).to_excel("Variables Check\\03.Variables_Post_Features_With_High_Nulls_Elimination.xlsx", sheet_name='Data', index=False);
    data=EliminateIdenticalFeatures(data); # improved over R code: additionally var53/var98 also removed
    pandas.DataFrame(list(data.columns), columns =['Var Names']).to_excel("Variables Check\\04.Variables_Post_Identical_Feature_Elimination.xlsx", sheet_name='Data', index=False);
    EliminateFeaturesWithHighZeroes(data, 90);
    pandas.DataFrame(list(data.columns), columns =['Var Names']).to_excel("Variables Check\\05.Variables_Post_Features_With_High_Zeroes_Elimination.xlsx", sheet_name='Data', index=False);
    EliminateFeaturesWithNearZeroVariance(data);
    pandas.DataFrame(list(data.columns), columns =['Var Names']).to_excel("Variables Check\\06.Variables_Post_Near_Zero_Variance_Feature_Elimination.xlsx", sheet_name='Data', index=False);
    EliminateFeaturesWithHighNonnumericUniqueValues(data, 20, 20);
    pandas.DataFrame(list(data.columns), columns =['Var Names']).to_excel("Variables Check\\07.Variables_Post_Features_With_High_Nonnumeric_Unique_Values_Elimination.xlsx", sheet_name='Data', index=False);
    EliminateCorrelatedFeatures(data, 0.85);
    pandas.DataFrame(list(data.columns), columns =['Var Names']).to_excel("Variables Check\\08.Variables_Post_Correlated_Feature_Elimination.xlsx", sheet_name='Data', index=False);
    # Variable Treatment
    MissingValueTreatment(data);
    OutlierTreatment(data);
    
    # Axis Selection
    option3=False;
    while not option3:
        try:
            axisnum=input("Enter the axis number (label) to be predicted:");
            data['ID']=rawframe['ID'].copy();
            data['FLAG']=rawframe['FLAG'].copy();
            data['Segment']=rawframe['L1'].copy();
            data['Segment2']=rawframe['L'+str(axisnum)].copy();
            data['Y']=rawframe['Y'+str(axisnum)].copy();
            option3=True;
        except AttributeError:
            print("There's no item with that code");
        except KeyError:
            print("Bad parameter name");
        except:
            print("Unknown error");
    # Bivariate Analysis - DEPRECATED
    featurelist=data.columns;
    numerics=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'];
    numfeaturelist=list(data.select_dtypes(include=numerics).columns);
    data_train=data.loc[data['FLAG']=='train'];
    #try: - DEPRECATED
        #os.mkdir('Bivariate Plots'); - DEPRECATED
    #except: - DEPRECATED
        #print("Address already exists.") - DEPRECATED
    #for feature in numfeaturelist: - DEPRECATED
        #scplot=matplotlib.pyplot.figure() - DEPRECATED
        #matplotlib.pyplot.scatter(x=data_train[feature], y=data_train['Y']); - DEPRECATED
        #_=scplot.savefig('Bivariate Plots\\'+str(feature)+' vs Y.pdf', bbox_inches='tight'); - DEPRECATED
        ##data_train.plot.scatter(x=feature, y='Y').savefig("\Plots\foo.pdf", bbox_inches='tight'); - DEPRECATED
    
    # Variable Transformation II - Normalization and Label-encoding
    data_normal=data.copy();
    sc=StandardScaler();# Normalization
    labelenc=LabelEncoder();# Label Encoding
    idlabelenc=LabelEncoder(); # ID Label Encoding
    featurelist=list(data_normal.columns);
    featurelist.pop();    # Not normalizing the Sales!
    for feature in featurelist:
        if data_normal[feature].dtype in numerics:
            data_normal[feature]=sc.fit_transform(data_normal[feature].values.reshape(-1,1));
        elif feature=='ID':
            data_normal[feature]=idlabelenc.fit_transform(data_normal[feature].astype(str));
        else:
            data_normal[feature]=labelenc.fit_transform(data_normal[feature].astype(str));
    
    #------------------------------VERIFIED------------------------------
    
    # Recursive Feature Elimination - NOT USED CURRENTLY
    data_normal['Segment2a']=data['Segment2'].copy();
    data_seg_train=data_normal.loc[data_normal['Segment2a'].notnull()];
    data_seg_test=data_normal.loc[data_normal['Segment2a'].isnull()];
    data_seg_train['Segment2b']=data_seg_train['Segment2'].copy();
    data_seg_id_label=data_seg_train[['ID','Y']];
    data_normal.drop(['Segment2a'], axis=1, inplace=True);
    data_seg_train.drop(['Segment2a','Segment2','Segment','ID','FLAG','Y'], axis=1, inplace=True);
    rf=RandomForestClassifier(n_estimators=25, n_jobs=-1, class_weight='balanced', max_depth=1, random_state=7);
    rfe=RFE(rf, 3, step=3, verbose=2);
    rfe.fit(data_seg_train.iloc[:,:len(data_seg_train.columns)-1],data_seg_train.iloc[:,len(data_seg_train.columns)-1]);
    rfeframe=pandas.DataFrame(columns=['Features','Result']);
    rfefeaturelist=list(data_seg_train.columns);
    rfefeaturelist.pop();    # Removing the Sales column
    rfefeaturearr=numpy.asarray(rfefeaturelist);
    rfeframe['Features']=rfefeaturearr;
    rfeframe['Result']=numpy.asarray(rfe.support_);
    rfeframe.loc[rfeframe['Result']==True].to_excel("Variables Check\\09.Variables_Post_Recursive_Feature_Elimination.xlsx", sheet_name='RFE', index=False);
    rfepassframe=rfeframe['Features'].loc[rfeframe['Result']==True];
    print(rfepassframe.to_string(index=False));
    
    # Boruta - USED CURRENTLY FOR FEATURE SELECTION
    while True:
        try:
            iters=int(input("Enter the num. of iterations for Boruta(recommended 50-100):"));
            if iters > 0:
                thres=int(input("Enter the threshold percent for Boruta(recommended 70-90):"));
                if 0 < thres < 50:
                    print("Threshold too low.");
                elif 50 <= thres <= 100:
                    break;
                elif thres > 100:
                    print("Threshold can't be over 100 percent.");
                else:
                    print("Invalid range for threshold input.");
            else:
                print("Num. of iterations cannot be zero/negative, re-enter a valid input...")
        except AttributeError:
            print("There's no item with that code");
        except KeyError:
            print("Bad parameter name");
        except:
            print("Unknown error");
    rf=RandomForestClassifier(n_estimators='auto', n_jobs=-1, class_weight='balanced', max_depth=3, random_state=7);
    boruta=BorutaPy(rf, n_estimators='auto', max_iter=iters, perc=thres, verbose=2, random_state=7);
    boruta.fit(data_seg_train.iloc[:,:len(data_seg_train.columns)-1].values, data_seg_train.iloc[:,len(data_seg_train.columns)-1].values);
    borutaframe=pandas.DataFrame(columns=['Features','Result']);
    borutafeaturelist=list(data_seg_train.columns);
    borutafeaturelist.pop();    # Removing the Sales column
    borutafeaturearr=numpy.asarray(borutafeaturelist);
    borutaframe['Features']=borutafeaturearr;
    while True:
        try:
            leniency=input("Consider 'Tentative' features for the model?[Y/N]");
            if leniency == 'Y' or leniency == 'y':
                borutaframe['Tentative']=numpy.asarray(boruta.support_weak_);
                borutaframe['Confirmed']=numpy.asarray(boruta.support_);
                #for i in range(len(borutaframe)):   # needs to be optimised
                #    borutaframe['Result'][i]=(borutaframe['Confirmed'][i] or borutaframe['Tentative'][i]);
                borutaframe['Result']=True;
                borutaframe['Result'].loc[borutaframe['Tentative'] == borutaframe['Confirmed']]=False;    #optimised
                break;
            elif leniency == 'N' or leniency == 'n':
                borutaframe['Result']=numpy.asarray(boruta.support_);
                break;
            else:
                print("Invalid input, please try again...");
        except AttributeError:
            print("There's no item with that code");
        except KeyError:
            print("Bad parameter name");
        except:
            print("Unknown error");
    borutaframe.loc[borutaframe['Result']==True].to_excel("Variables Check\\10.Variables_Post_Boruta_Feature_Elimination.xlsx", sheet_name='Boruta', index=False);
    borutapassframe=borutaframe['Features'].loc[borutaframe['Result']==True];
    print(borutapassframe.to_string(index=False));
    
    # Pre-Modelling - Master Train, Test and Validation sets
    xv_data=pandas.concat([data_seg_train, data_seg_id_label], axis=1, ignore_index=False);
    numofseg=len(collections.Counter(xv_data['Segment2b']));
    x_train, x_test, y_train, y_test = train_test_split(xv_data.iloc[:,:len(xv_data.columns)-1].drop(['Segment2b','ID'],axis=1), \
                                                                    xv_data.iloc[:,len(xv_data.columns)-3], \
                                                                    test_size=round(0.2*len(xv_data)/numofseg)*numofseg/len(xv_data), \
                                                                    random_state=7, stratify=xv_data['Segment2b']);
    x_overall=pandas.concat([x_train, x_test]);
    y_overall=pandas.concat([y_train, y_test]);    # Label is Segment
    #x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(xv_data.iloc[:,:len(xv_data.columns)-1].drop(['Segment2b','ID'],axis=1), \
    #                                                                         data_train.iloc[:,len(data_train.columns)-1], \
    #                                                                         test_size=round(0.2*len(xv_data)/numofseg)*numofseg/len(xv_data), \
    #                                                                         random_state=7);    # Removed: stratify=xv_data['Segment2b']
    #x_overall_2=pandas.concat([x_train_2, x_test_2]);
    #y_overall_2=pandas.concat([y_train_2, y_test_2]);    # Label is Sales
    
    # Feature Importance, Selection & Iterations
    register_template=pandas.DataFrame(columns=['Iter','Model','F1_Self','Accu_Self','F1_Valid','Accu_Valid','F1_Overall','Accu_Overall','Selected Var','Distro-Self','Distro-Valid','Distro-Overall','Distro-Universe'], \
                                       data=[[0,'NIL',0,0,0,0,0,0,'NIL','NIL','NIL','NIL','NIL']]);
    register=register_template[0:0].copy();
    selection=True;
    iteration=0;
    distro='';
    templist=list(y_train);
    for element in set(y_train):
        distro+=(str(templist.count(element))+'\t');
    print("Original Distribution:"+distro);
    newfeaturelist=list(borutapassframe);
    verifiedfeaturelist=[];
    exemptfeaturelist=[];
    while selection:
        iteration+=1;
        xs_train=x_train[newfeaturelist].copy();
        xs_test=x_test[newfeaturelist].copy();
        xs_overall=x_overall[newfeaturelist].copy();
        univ=data_normal[newfeaturelist].copy();
        selected_var='';
        for var in newfeaturelist:
            selected_var+=(str(var)+' + ');
        rf=RandomForestClassifier(n_estimators=50, n_jobs=-1, class_weight='balanced', max_depth=10, random_state=7);
        rf.fit(xs_train, y_train);
        model='RF';
        y_pred_self=rf.predict(xs_train);
        f1_self=f1_score(y_train, y_pred_self, average='macro');
        accu_self=accuracy_score(y_train, y_pred_self);
        distro_self='';
        templist=list(y_pred_self);
        for element in set(y_pred_self):
            distro_self+=(str(templist.count(element))+'\t');
        y_pred_valid=rf.predict(xs_test);
        f1_valid=f1_score(y_test, y_pred_valid, average='macro');
        accu_valid=accuracy_score(y_test, y_pred_valid);
        distro_valid='';
        templist=list(y_pred_valid);
        for element in set(y_pred_valid):
            distro_valid+=(str(templist.count(element))+'\t');
        y_pred_overall=rf.predict(xs_overall);
        f1_overall=f1_score(y_overall, y_pred_overall, average='macro');
        accu_overall=accuracy_score(y_overall, y_pred_overall);
        distro_overall='';
        templist=list(y_pred_overall);
        for element in set(y_pred_overall):
            distro_overall+=(str(templist.count(element))+'\t');
        univ_pred=rf.predict(univ);
        distro_univ='';
        templist=list(univ_pred);
        for element in set(univ_pred):
            distro_univ+=(str(templist.count(element))+'\t');
        # Update Register
        record=register_template.copy();
        record['Iter']=iteration;
        record['Model']=model;
        record['F1_Self']=f1_self;
        record['Accu_Self']=accu_self;
        record['F1_Valid']=f1_valid;
        record['Accu_Valid']=accu_valid;
        record['F1_Overall']=f1_overall;
        record['Accu_Overall']=accu_overall;
        record['Selected Var']=selected_var;
        record['Distro-Self']=distro_self;
        record['Distro-Valid']=distro_valid;
        record['Distro-Overall']=distro_overall;
        record['Distro-Universe']=distro_univ;
        register=register.append(record);
        # Run Registered
        sm=SMOTE(random_state=7);
        xs_synth, y_synth = sm.fit_sample(xs_train, y_train); 
        rf.fit(xs_synth, y_synth);
        model='RF+S';
        y_pred_self=rf.predict(xs_train);
        f1_self=f1_score(y_train, y_pred_self, average='macro');
        accu_self=accuracy_score(y_train, y_pred_self);
        distro_self='';
        templist=list(y_pred_self);
        for element in set(y_pred_self):
            distro_self+=(str(templist.count(element))+'\t');
        y_pred_valid=rf.predict(xs_test);
        f1_valid=f1_score(y_test, y_pred_valid, average='macro');
        accu_valid=accuracy_score(y_test, y_pred_valid);
        distro_valid='';
        templist=list(y_pred_valid);
        for element in set(y_pred_valid):
            distro_valid+=(str(templist.count(element))+'\t');
        y_pred_overall=rf.predict(xs_overall);
        f1_overall=f1_score(y_overall, y_pred_overall, average='macro');
        accu_overall=accuracy_score(y_overall, y_pred_overall);
        distro_overall='';
        templist=list(y_pred_overall);
        for element in set(y_pred_overall):
            distro_overall+=(str(templist.count(element))+'\t');
        univ_pred=rf.predict(univ);
        distro_univ='';
        templist=list(univ_pred);
        for element in set(univ_pred):
            distro_univ+=(str(templist.count(element))+'\t');
        # Update Register
        record=register_template.copy();
        record['Iter']=iteration;
        record['Model']=model;
        record['F1_Self']=f1_self;
        record['Accu_Self']=accu_self;
        record['F1_Valid']=f1_valid;
        record['Accu_Valid']=accu_valid;
        record['F1_Overall']=f1_overall;
        record['Accu_Overall']=accu_overall;
        record['Selected Var']=selected_var;
        record['Distro-Self']=distro_self;
        record['Distro-Valid']=distro_valid;
        record['Distro-Overall']=distro_overall;
        record['Distro-Universe']=distro_univ;
        register=register.append(record);
        # Run Registered
        ETC=ExtraTreesClassifier();    # Calculates Feature Importances
        ETC.fit(xs_train, y_train);
        x_imp_a=numpy.array(xs_train.columns);
        x_imp_b=ETC.feature_importances_;
        var_imp=pandas.DataFrame({'var':x_imp_a,'imp':x_imp_b});
        var_imp.sort_values(by=['imp'], ascending=True, inplace=True);
        if len(xs_train.columns) > 0.1*len(x_train.columns) \
        and len(xs_train.columns) > 7:    #Final selection should not have more than 7 variables
        #and len(newfeaturelist) > len(verifiedfeaturelist):
            if iteration == 1:
                best_iter=iteration;
                best_f1=max(list(register['F1_Self']));
                newfeaturelist.remove(var_imp['var'].iloc[0]); # Removing least important var
                featureofinterest=var_imp['var'].iloc[0];
            else:
                if max(list(register['F1_Self'].iloc[len(register)-2:])) > best_f1:
                    best_iter=iteration;    # F1 Score Improved
                    best_f1=max(list(register['F1_Self']));
                    exemptfeaturelist.append(featureofinterest);
                    newfeaturelist.remove(var_imp['var'].iloc[0]); # Removing least important var
                    featureofinterest=var_imp['var'].iloc[0];
                else:
                    if len(newfeaturelist) > len(verifiedfeaturelist):
                        verifiedfeaturelist.append(featureofinterest);    # F1 Score declined
                        newfeaturelist.append(featureofinterest);    # Keep the feature of interest
                        for feature in verifiedfeaturelist:
                            var_imp=var_imp[var_imp['var'] != feature];
                        var_imp.sort_values(by=['imp'], ascending=True, inplace=True);
                        newfeaturelist.remove(var_imp['var'].iloc[0]); # Removing least important var
                    else:
                        newfeaturelist.remove(var_imp['var'].iloc[0]);
                    featureofinterest=var_imp['var'].iloc[0];
        else:
            selection=False;
        print('Iteration '+str(iteration)+' done.');
    register.to_excel("Register.xlsx", sheet_name='Register', index=False);
    print("******* Check the Register.xlsx generated to choose the best model! *******")
    #EXTRAS: Use the below code to get extra info from your runs after te end of program
    #print('Confusion Matrix: [Overall Data - Self]\n'+str(confusion_matrix(y_overall, y_pred_overall)));
    #print('Accuracy Score : [Overall Data - Self]\n'+str(round(100*accuracy_score(y_overall, y_pred_overall), 2))+'%');
    #print('Report : [Overall Data - Self]\n'+classification_report(y_overall, y_pred_overall));
    
    # Parameter Selection and Model Predictions (Finale)
    option=False;
    while not option:
        try:
            rfc=RandomForestClassifier(n_estimators=50, n_jobs=-1, class_weight='balanced', max_depth=10, random_state=7);
            rfr=RandomForestRegressor(n_estimators=50, max_depth=10, random_state=7);
            modelframe=pandas.DataFrame(columns=[]);
            testvarlist=[];
            testvartotal=input("Enter the number of variables to be entered into the model:");
            for i in range(int(testvartotal)):
                testvarname=input("Enter variable "+str(i+1)+"'s name:");
                modelframe[testvarname]=data_normal[testvarname].copy(); #changed
                testvarlist.append(testvarname);
            predictor=input("(RF sales prediction deprecated in current version)\nEnter 'S' to predict segment or Enter 'Y' to predict sales:");
            if 'S' in predictor or 's' in predictor:
                option3=True;
                while option3:
                    synthesize=input("Do you wish to synthesize samples (class imbalance) for training?[Y/N]");
                    if synthesize=='Y' or synthesize=='y':
                        sm=SMOTE(random_state=7);
                        x_synth, y_synth = sm.fit_sample(x_train[testvarlist], y_train);
                        rfc.fit(x_synth, y_synth);
                        option3=False;
                    elif synthesize=='N' or synthesize=='n':
                        rfc.fit(x_train[testvarlist], y_train);
                        option3=False;
                predictionarray=rfc.predict(modelframe.values);
                labelenc_map_dict=dict(zip(labelenc.classes_, labelenc.transform(labelenc.classes_)));
                inverted_map_dict = dict(map(reversed, labelenc_map_dict.items()));
                predictionarray2=numpy.array([inverted_map_dict[letter] for letter in predictionarray]);
                predictionframe=pandas.DataFrame(columns=[]);
                id_labelenc_map_dict=dict(zip(idlabelenc.classes_, idlabelenc.transform(idlabelenc.classes_)));
                id_inverted_map_dict = dict(map(reversed, id_labelenc_map_dict.items()));
                idarray=data_normal['ID'].values; #changed
                predictionframe['ID']=numpy.array([id_inverted_map_dict[letter] for letter in idarray]);
                predictionframe['Predicted Segment']=predictionarray2;
                predictionframe.to_excel("Segment_Prediction.xlsx", sheet_name='RandomForestModel', index=False);
                print("******* The predictions have been exported to Segment_Prediction.xlsx *******");
                option2=True;
                while option2:
                    retry=input("Do you wish to try another set of test variables?[Y/N]");
                    if retry=='Y' or retry=='y':
                        option=False;
                        option2=False;
                    elif retry=='N' or retry=='n':
                        option=True;
                        option2=False;
                    else:
                        option2=True;
            elif 'Y' in predictor or 'y' in predictor:
                rfr.fit(x_overall_2[testvarlist], y_overall_2);
                predictionarray=rfr.predict(modelframe.values);
                predictionframe=pandas.DataFrame(columns=[]);
                id_labelenc_map_dict=dict(zip(idlabelenc.classes_, idlabelenc.transform(idlabelenc.classes_)));
                id_inverted_map_dict = dict(map(reversed, id_labelenc_map_dict.items()));
                idarray=data_normal['ID'].values; #changed
                predictionframe['ID']=numpy.array([id_inverted_map_dict[letter] for letter in idarray]);
                predictionframe['Predicted Sales']=predictionarray;
                predictionframe.to_excel("Sales_Prediction.xlsx", sheet_name='RandomForestModel', index=False);
                option2=True;
                while option2:
                    retry=input("Do you wish to try another set of test variables?[Y/N]");
                    if retry=='Y' or retry=='y':
                        option=False;
                        option2=False;
                    elif retry=='N' or retry=='n':
                        option=True;
                        option2=False;
                    else:
                        option2=True;
            else:
                print("Invalid input. Try again.");
        except AttributeError:
            print("There's no item with that code");
        except KeyError:
            print("Bad parameter name");
        except:
            print("Unknown error");
    
    
    #-----------------------------WORKSPACE-----------------------------