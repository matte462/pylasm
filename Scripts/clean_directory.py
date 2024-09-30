import os

src_dir = '.'
for subdir in os.listdir(src_dir) :
    if os.path.isdir(subdir) :
        for file in os.listdir(subdir) :
            
            # SPIN_OUT.json and SPIN_REPORT.txt contain all the data for post-processing
            os.chdir(subdir)
            if file not in ['SPIN_OUT.json','SPIN_REPORT.txt']:
                os.remove(file)
            os.chdir('..')