import os
import sys
import pandas

def LineGroup(journeytree, linenum, linename, subframe):
    #init
    if linename == '' and linenum != 0:
        return journeytree;
    tempframe=subframe.copy();
    linelist=list(subframe.columns);
    if linenum != 0:
        linelist.remove('Line'+str(linenum-1));
    tempframe=tempframe[linelist].copy();
    linelist.remove('Total');
    #subset creation
    tempframe=tempframe.loc[tempframe['Line'+str(linenum)]==linename].copy();
    #setting cut-offs - to be confirmed
    #tempframe=tempframe[tempframe['Total'] >= 0.05*tempframe['Total'].sum()];
    #roll-up
    tempframe=tempframe.groupby(linelist).agg({'Total':'sum'}).reset_index(drop=False);
    #update journeytree
    journeytree+='vt.Node("'+linename+'(n='+str(tempframe['Total'].sum())+')")([';
    #continue chain
    sublinelist=set(list(tempframe['Line'+str(linenum+1)]));
    linenum+=1;
    for sublinename in sublinelist:
        journeytree=LineGroup(journeytree, linenum, sublinename, tempframe);
    if linename == '' and linenum != 0:
        journeytree+='])';
    else:
        journeytree+=']), ';
    return journeytree;

werkdir=os.path.realpath(__file__);
werkdir=werkdir.split('PatientJourneyTracer.py')[0];
journeyframe=pandas.read_csv(werkdir+"//Lines.csv");
journeyframe.fillna("", inplace=True);
journeyframe['Line0']='';
numoflines=len(list(journeyframe.columns))-1;
journeyframe['Line'+str(numoflines)]="";
journeytree='import VisualTreeRosettaCodeDotOrg as vt\njourneytree=';
journeytree=LineGroup(journeytree, 0, '', journeyframe);
journeytree+="\nprint('\\n\\n'.join([vt.drawTree2(False)(False)(journeytree)]));";
#journeytree=journeytree[:-2];
journeytreedotpy=open(werkdir+"//JourneyTree.py",'w');
journeytreedotpy.write(journeytree);
journeytreedotpy.close();
original_stdout=sys.stdout;
outputfile=open('Journey.txt', 'w', encoding='utf-8');
sys.stdout=outputfile;

import JourneyTree

sys.stdout=original_stdout;
outputfile.close();