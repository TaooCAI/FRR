import os
import os.path
import torch

IMG_EXTENSIONS = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    '.png',
    '.PNG',
    '.ppm',
    '.PPM',
    '.bmp',
    '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def __absolute2relative(data):
    for i, item in enumerate(data):
        for j, path in enumerate(item):
            s_path = path.split('/')
            label = list(filter(lambda x:'monkaa' in x.lower() or 'driving' in x.lower() or 'flying' in x.lower(),s_path))[0]
            data[i][j] = '/'.join(s_path[s_path.index(label):])
    return data


def __relative2absolute(data, prefix):
    for i, item in enumerate(data):
        for j, path in enumerate(item):
            data[i][j] = os.path.join(prefix, path)
    return data


def dataloader(filepath):
    store_path = './flying3d_relative_pathlist_tuple_except_unused_files.pth'
    if os.path.exists(store_path):
        print('Loading dataset from {}'.format(store_path))
        store_data = torch.load(store_path)['data']
        return __relative2absolute(store_data, filepath)

    import sys
    print('The dataset is not flyingthings3D that I need.', file=sys.stderr)
    sys.exit(-1)

    classes = [
        d for d in os.listdir(filepath)
        if os.path.isdir(os.path.join(filepath, d))
    ]
    image = [img for img in classes if img.find('frames_cleanpass') > -1]
    disp = [dsp for dsp in classes if dsp.find('disparity') > -1]

    monkaa_path = filepath + [x for x in image if 'monkaa' in x][0]
    monkaa_disp = filepath + [x for x in disp if 'monkaa' in x][0]

    monkaa_dir = os.listdir(monkaa_path)

    all_left_img = []
    all_right_img = []
    all_left_disp = []
    test_left_img = []
    test_right_img = []
    test_left_disp = []

    for dd in monkaa_dir:
        for im in os.listdir(monkaa_path + '/' + dd + '/left/'):
            if is_image_file(monkaa_path + '/' + dd + '/left/' + im):
                all_left_img.append(monkaa_path + '/' + dd + '/left/' + im)
                all_left_disp.append(monkaa_disp + '/' + dd + '/left/' +
                                     im.split(".")[0] + '.pfm')

        for im in os.listdir(monkaa_path + '/' + dd + '/right/'):
            if is_image_file(monkaa_path + '/' + dd + '/right/' + im):
                all_right_img.append(monkaa_path + '/' + dd + '/right/' + im)

    flying_path = filepath + [x for x in image if 'flyingthings3d' in x][0]
    flying_disp = filepath + [x for x in disp if 'flyingthings3d' in x][0]
    flying_dir = flying_path + '/TRAIN/'
    subdir = ['A', 'B', 'C']

    for ss in subdir:
        flying = os.listdir(flying_dir + ss)

        for ff in flying:
            imm_l = os.listdir(flying_dir + ss + '/' + ff + '/left/')
            for im in imm_l:
                if is_image_file(flying_dir + ss + '/' + ff + '/left/' + im):
                    all_left_img.append(flying_dir + ss + '/' + ff + '/left/' +
                                        im)

                all_left_disp.append(flying_disp + '/TRAIN/' + ss + '/' + ff +
                                     '/left/' + im.split(".")[0] + '.pfm')

                if is_image_file(flying_dir + ss + '/' + ff + '/right/' + im):
                    all_right_img.append(flying_dir + ss + '/' + ff +
                                         '/right/' + im)

    flying_dir = flying_path + '/TEST/'

    subdir = ['A', 'B', 'C']

    for ss in subdir:
        flying = os.listdir(flying_dir + ss)

        for ff in flying:
            imm_l = os.listdir(flying_dir + ss + '/' + ff + '/left/')
            for im in imm_l:
                if is_image_file(flying_dir + ss + '/' + ff + '/left/' + im):
                    test_left_img.append(flying_dir + ss + '/' + ff +
                                         '/left/' + im)

                test_left_disp.append(flying_disp + '/TEST/' + ss + '/' + ff +
                                      '/left/' + im.split(".")[0] + '.pfm')

                if is_image_file(flying_dir + ss + '/' + ff + '/right/' + im):
                    test_right_img.append(flying_dir + ss + '/' + ff +
                                          '/right/' + im)

    driving_dir = filepath + [x for x in image if 'driving' in x][0] + '/'
    driving_disp = filepath + [x for x in disp if 'driving' in x][0]

    subdir1 = ['15mm_focallength', '35mm_focallength']
    subdir2 = ['scene_backwards', 'scene_forwards']
    subdir3 = ['fast', 'slow']

    for i in subdir1:
        for j in subdir2:
            for k in subdir3:
                imm_l = os.listdir(driving_dir + i + '/' + j + '/' + k +
                                   '/left/')
                for im in imm_l:
                    if is_image_file(driving_dir + i + '/' + j + '/' + k +
                                     '/left/' + im):
                        all_left_img.append(driving_dir + i + '/' + j + '/' +
                                            k + '/left/' + im)
                    all_left_disp.append(driving_disp + '/' + i + '/' + j +
                                         '/' + k + '/left/' +
                                         im.split(".")[0] + '.pfm')

                    if is_image_file(driving_dir + i + '/' + j + '/' + k +
                                     '/right/' + im):
                        all_right_img.append(driving_dir + i + '/' + j + '/' +
                                             k + '/right/' + im)

    all_data = (all_left_img, all_right_img, all_left_disp, test_left_img,
                test_right_img, test_left_disp)
    import copy
    save_data = __absolute2relative(copy.deepcopy(all_data))
    store_data = {'data': save_data}
    torch.save(store_data, store_path)

    return all_data


if __name__ == '__main__':
    dataloader('/home/caitao/Dataset/sceneflow/')
