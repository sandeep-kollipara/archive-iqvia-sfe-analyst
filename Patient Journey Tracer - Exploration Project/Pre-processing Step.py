import pandas

class ExitScript(Exception): pass;

try:
    fileaddress=input("Enter the address of the .csv file with patient lines:");
    inputframe=pandas.read_csv(fileaddress);
    cutoff=float(input("Enter the cutoff for the high patient lines(between 0 & 1):"));
    if cutoff > 1 or cutoff < 0:
        raise ExitScript("Parameter out of bounds.");
except AttributeError:
    print("There's no item with that code");
except KeyError:
    print("Bad parameter name");
except:
    print("Unknown error");
intermediateframe=inputframe.groupby('Line1').agg({'Total':'sum'}).reset_index(drop=False);
intermediateframe.sort_values(by=['Total'], ascending=False, inplace=True);
intermediateframe['Cumulative']=intermediateframe['Total'].cumsum();
cumcutoff=cutoff*intermediateframe['Cumulative'].max();
intermediateframe=intermediateframe.loc[intermediateframe['Cumulative'] < cumcutoff];
if len(intermediateframe) == 0:
    raise ExitScript("No lines remain after applying the cutoff.");
intermediateframe.drop(columns=['Total', 'Cumulative'], inplace=True);
outputframe=pandas.merge(inputframe, intermediateframe, how='right', on='Line1');
outputframe.sort_values(by=['Total'], ascending=False, inplace=True);
outputframe.to_csv(fileaddress[:-4]+'_proc.csv', index=False);