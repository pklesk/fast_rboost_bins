Actual contents of this folder is not stored in the github repository due to large size of FDDB data. 

To reproduce the needed contents, download and unzip here: 
http://vis-www.cs.umass.edu/fddb/originalPics.tar.gz
http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz

Then, actual data sets (data arrays) can be generated from /data_raw folder into /data folder as pickled .bin files.
This can be done by using /src/main_detector.py script with data regeneration option (-rd or --REGENERATE_DATA) and other wanted settings.  