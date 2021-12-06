import os
import zipfile
from utils.augmentation import Compose, Scale, RandomMirror, RandomRotation, Resize, Normalize_Tensor
from torch.utils import data
from PIL import Image
import matplotlib.pyplot as plt

def unzip_data(data_dir):
    path_file_zip = os.path.join(data_dir, 'archive.zip')
    if not os.path.exists(os.path.join(data_dir, 'VOC2012')):
        with zipfile.ZipFile(path_file_zip, 'r') as zip_file:
            zip_file.extractall(data_dir)


def make_data_path_list(root_path):
    original_image_template = os.path.join(root_path, 'JPEGImages', '%s.jpg')
    annotation_image_template = os.path.join(root_path, 'SegmentationClass', '%s.png')

    # train, val
    train_ids = os.path.join(root_path, 'ImageSets/Segmentation/train.txt')
    val_ids = os.path.join(root_path, 'ImageSets/Segmentation/val.txt')

    train_img_list = list()
    train_anno_list = list()

    for line in open(train_ids):
        img_id = line.strip()
        img_path = (original_image_template % img_id)
        anno_path = (annotation_image_template % img_id)

        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

    val_img_list = list()
    val_anno_list = list()

    for line in open(val_ids):
        img_id = line.strip()
        img_path = (original_image_template % img_id)
        anno_path = (annotation_image_template % img_id)

        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list





class DataTransform():
    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            "train": Compose([
                Scale(scale=[0.5, 1.5]),
                RandomRotation(angle=[-10, 10]),
                RandomMirror(),
                Resize(input_size),
                Normalize_Tensor(color_mean, color_std)
            ]),
            "val": Compose([
                Resize(input_size),
                Normalize_Tensor(color_mean, color_std)
            ])
        }

    def __call__(self, phase, img, anno_class_img):
        # phase: training or validation
        return self.data_transform[phase](img, anno_class_img)

class MyDataSet(data.Dataset):
    def __init__(self, img_list, anno_list, phase, transform):
        self.img_list= img_list
        self.anno_list= anno_list
        self.phase= phase
        self.transform= transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img, anno_class_img= self.pull_item(item)
        return  img, anno_class_img

    def pull_item(self, index):
        # original image
        img_file_path= self.img_list[index]
        img= Image.open(img_file_path)

        #annotation image
        anno_file_path= self.anno_list[index]
        anno_class_img= Image.open(anno_file_path)

        img, anno_class_img= self.transform(self.phase, img, anno_class_img)

        return img, anno_class_img



def main():
    data_dir = './data'
    unzip_data(data_dir)
    root_path = os.path.join(data_dir, 'VOC2012')
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_data_path_list(root_path)
    #
    # print(len(train_img_list))
    # print(len(val_img_list))

    color_mean=(0.485, 0.456, 0.406)
    color_std=(0.229, 0.224, 0.225)

    train_dataset= MyDataSet(train_img_list, train_anno_list, phase='train', transform= DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))
    val_dataset= MyDataSet(val_img_list, val_anno_list, phase='val', transform= DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))

    # print('val dataset img: {}'.format(val_dataset.__getitem__(0)[0].shape))
    # print('val dataset anno class img: {}'.format(val_dataset.__getitem__(0)[1].shape))
    # print('val dataset: {}'.format(val_dataset.__getitem__(0)))

    batch_size=4
    train_data_loader= data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader= data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    data_loader_dict={
        'train': train_data_loader,
        'val': val_data_loader
    }
    # tao ra mot cum cac anh
    batch_iterator= iter(data_loader_dict['train'])

    # lay ra cac cum, moi cum gom 4 anh goc, 4 anh annotation
    images, anno_class_images= next(batch_iterator)

    # print(images.size())
    # print(anno_class_images.size())

    image= images[0].numpy().transpose(1,2,0) #(chanel(RGB), height, witdh) => (height, width, chanel(RGB))
    plt.imshow(image)
    plt.show()

    anno_class_image= anno_class_images[0].numpy()
    plt.imshow(anno_class_image)
    plt.show()


if __name__ == '__main__':
    main()
