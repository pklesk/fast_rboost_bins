Actual contents of this folder is not stored in the github repository due to large size of HaGRID data (repository: https://github.com/hukenovs/hagrid).

To reproduce the needed contents, download and unzip here the subset of images with "palm" gesture:
https://n-usr-2uzac.s3pd02.sbercloud.ru/b-usr-2uzac-mv4/hagrid/train_val_palm.zip

Then, actual data sets (data arrays) can be generated from /data_raw folder into /data folder as pickled .bin files.
This can be done by using /src/main_detector.py script with data regeneration option (-rd or --REGENERATE_DATA) and other wanted settings.