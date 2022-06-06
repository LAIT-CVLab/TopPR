import numpy as np
import os
###########################################
#       Save Numpy Datas
###########################################
def save_numpy(np_data, path, name):
    """
    numpy를 npy 파일 형태로 저장하는 함수
    :np_data: data
    :path:    folder path
    :name:    파일명
    """
    if not isinstance(np_data, np.ndarray):
        np_data = np.asarray(np_data)
        path = os.path.join(path, name + '.npy')
        np.save(np_data, path)
        print('file saving is successed ', path)
    else:
        print('file saving is failed')

def save_npdict(npdict_data, path, name):
    """
    numpy를 npz 파일 형태로 저장하는 함수
    :npdict_data: data
    :path:    folder path
    :name:    파일명
    """
    if not isinstance(npdict_data, dict):
        npdict_data = dict(npdict_data)
        path = os.path.join(path, name + '.npz')
        np.savez(npdict_data, path)
        print('file saving is successed ', path)
    else:
        print('file saving is failed')


###############################################
#       Make Experiment Results Directory 
###############################################
def make_sub_results_dirs(results_path, args,):
    dirlist = []
    if args.truncation_trick_toy:
        dirlist.append('truncation_trick_toy')
        
    if args.truncation_trick_real_data:
        dirlist.append('truncation_trick_real_data')
        
    if args.simultaneous_mode_drop_toy:
        dirlist.append('simultaneous_mode_drop_toy')
        
    if args.simultaneous_mode_drop_real_data:
        dirlist.append('simultaneous_mode_drop_real_data')
        
    if args.sequential_mode_drop_toy:
        dirlist.append('sequential_mode_drop_toy')
        
    if args.sequential_mode_drop_real_data :
        dirlist.append('sequential_mode_drop_real_data')
        
    if args.inoutlier_toy :
        dirlist.append('inoutlier_toy')
        
    if args.inoutlier_real_data :
        dirlist.append('inoutlier_real_data')
        
    if args.realism_score_real_data :
        dirlist.append('realism_score_real_data')
        
    if args.single_score :
        dirlist.append('single_score')
        
    if args.per_sample_score :
        dirlist.append('per_sample_score')
        
    if args.check_outlier :
        dirlist.append('per_sample_score')
        
    [os.makedir(os.path.join(results_path, sub_path)) for sub_path in dirlist]
    

def folder_formatting(num):
    num_str = "%05d" % num
    return num_str

def numbering_dir(results_path):
    """
    directory를 생성할 때 자동으로 numbering을 한다.
    :results_path: results를 저장할 상위 폴더
    """
    dir_list = sorted(os.listdir(results_path))
    if dir_list:
        dir_num = int(dir_list[-1].split('_')[-1])
    else:
        dir_num = 0
    return dir_num


# To do 
def make_main_results_dirs(main_path, args, desc=None):

    os.makedirs(main_path, exist_ok=True)   # make  main_path/ directory

    dir_num = numbering_dir(main_path) + 1
    dir_num = folder_formatting(dir_num)
    if desc is None:
        desc = ""
    dir_name = "results" + desc + "_" + dir_num
    results_path = os.path.join(main_path, dir_name)
    os.makedirs(results_path, exist_ok=True)    # main_path/results{desc}_00001

    make_sub_results_dirs(results_path, args)


########################################################
#       Get Current Experiment Results Directory 
########################################################
def get_current_result_dir(results_path):
    dir_num = numbering_dir(results_path)
    return dir_num