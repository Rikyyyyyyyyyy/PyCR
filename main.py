import PyCR


# isexternal,howMuchSplit,isMicro,tupaType,isMotabo,MotaboFileName,DataFileName,ClassFileName,sampleNameFile,variableNameFile
## Tupa Selection: tupa, classtupa, notupa
## Scale Selection: SNV,AutoScale
#main(True,0.5,False,'notupa',False,'Input/mota_data.csv','Input/data_pureOil.csv','Input/class_pureOil_2.csv','Input/sampleName_pureOil.csv','Input/Vname_pureOil.csv','AutoScale',10,0.85)
#main(True,0.5,False,'classTupa',False,'Input/mota_data.csv','Input/data_algae.csv','Input/class_algae_string.csv','Input/S_name.csv','Input/v_name.csv','AutoScale',10,0.85)

## Needed parameters
## 1 is external
## 2 the rate of splite the trainning and validation
## 3 is Micro for ROC
## 4 What kind of TUPA : a) classtupa b)tupa c)notupa
## 5 is the input data from motabo analize
## 6 your motabo data file name, if not using motabo data just input None instead
## 7 your data file name (not motabo data )
## 8 your class file name (not motabo data )
## 9 your sample name file name (not motabo data )
## 10 your variable name file name (not motabo data )
## 11 how would you like to scale your data: a) AotuScale b)SNV
## 12 how many iterations you like
## 13 the survival rate

def main():

    # PyCR.runPyCR('true',0.34,'false','classTupa','false','','input/data_algae.csv', 'input/class_algae_string.csv', 'input/S_name.csv', 'input/v_name.csv', 'AutoScale', 1,2,'selectivity',3)
    #PyCR.runPyCR('true',0.34,'false','classTupa','false','','input/algae/data_algae.csv', 'input/algae/class_algae_string.csv', 'input/algae/S_name.csv', 'input/algae/v_name.csv', 'AutoScale', 100,0.85,'selectivity',3)
    PyCR.runPyCR('true',0.34,'false','classTupa','false','','input/coffee/data_coffee.csv', 'input/coffee/class_coffee.csv', 'input/coffee/Sname_coffee.csv', 'input/coffee/Vname_coffee.csv', 'AutoScale', 5,0.85,'selectivity',3)
    # PyCR.runPyCR('true',0.34,'false','notupa','false','','input/PANCAN/data_PANCAN.csv', 'input/PANCAN/labels_PANCAN.csv', 'input/PANCAN/Sname_PANCAN.csv', 'input/PANCAN/Vname_PANCAN.csv', 'AutoScale', 3,0.85,'selectivity',3)



main()