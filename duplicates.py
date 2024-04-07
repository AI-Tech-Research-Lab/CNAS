from cnas import CNAS

#create a kwargs arg for cnas
kwargs = {'first_obj': 'top1', 'sec_obj':'params', 'resume': '../results/nasbench/iter_12'}
cnas = CNAS(kwargs)
archive = cnas._resume_from_dir()
print(archive[-10:])


