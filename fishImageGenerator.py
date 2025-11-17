import os
import random
import time
import warnings
from math import pi, cos, sin
import cv2
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)


class FishEyeGenerator:

    def __init__(self, focal_len, dst_shape):

        self._focal_len = focal_len
        self._shape = dst_shape
        self._ratio = min(self._shape[0], self._shape[1]) / (self._focal_len * pi)

        mask = np.ones([self._shape[0], self._shape[1]], dtype=np.uint8)
        square_r = (min(self._shape[0], self._shape[1]) / 2) ** 2
        for i in range(self._shape[0]):
            for j in range(self._shape[1]):
                if ((i - self._shape[0] / 2) ** 2 + (j - self._shape[1] / 2) ** 2) >= square_r:
                    mask[i, j] = 0
        mask = np.array(mask)
        mask = mask.reshape(-1)
        self._bad_index = (mask == 0)

        self._PARAM = 500
        self._init_ext_params()

        self._bkg_color = [0, 0, 0]
        self._bkg_label = 19

    def set_bkg(self, bkg_label=19, bkg_color=[0, 0, 0]):
        self._bkg_color = list(bkg_color)
        self._bkg_label = bkg_label

    def _init_ext_params(self):
        self.ALPHA_RANGE = [0, 0]
        self.BETA_RANGE = [0, 0]
        self.THETA_RANGE = [0, 0]

        self.XTRANS_RANGE = [-self._shape[1] / 2, self._shape[1] / 2]
        self.YTRANS_RANGE = [-self._shape[0] / 2, self._shape[0] / 2]
        self.ZTRANS_RANGE = [-0.6 * self._PARAM, 0.6 * self._PARAM]

        self._alpha = 0
        self._beta = 0
        self._theta = 0
        self._x_trans = 0
        self._y_trans = 0
        self._z_trans = 0

    def set_ext_param_range(self, ext_param_range):
        self.ALPHA_RANGE = [-ext_param_range[0] * pi / 180, ext_param_range[0] * pi / 180]
        self.BETA_RANGE = [-ext_param_range[1] * pi / 180, ext_param_range[1] * pi / 180]
        self.THETA_RANGE = [-ext_param_range[2] * pi / 180, ext_param_range[2] * pi / 180]

        self.XTRANS_RANGE = [-self._shape[1] * ext_param_range[3], self._shape[1] * ext_param_range[3]]
        self.YTRANS_RANGE = [-self._shape[0] * ext_param_range[4], self._shape[0] * ext_param_range[4]]
        self.ZTRANS_RANGE = [-ext_param_range[5] * self._PARAM, ext_param_range[5] * self._PARAM]

    def _init_ext_matrix(self):
        self._rotate_trans_matrix = \
            np.array([
                [cos(self._beta) * cos(self._theta), cos(self._beta) * sin(self._theta), -sin(self._beta),
                 self._x_trans],
                [-cos(self._alpha) * sin(self._theta) + sin(self._alpha) * sin(self._beta) * cos(self._theta),
                 cos(self._alpha) * cos(self._theta) + sin(self._alpha) * sin(self._beta) * sin(self._theta),
                 sin(self._alpha) * cos(self._beta), self._y_trans],
                [sin(self._alpha) * sin(self._theta) + cos(self._alpha) * sin(self._beta) * cos(self._theta),
                 -sin(self._alpha) * cos(self._theta) + cos(self._alpha) * sin(self._beta) * sin(self._theta),
                 cos(self._alpha) * cos(self._beta), self._z_trans],
                [0, 0, 0, 1]
            ])

    def set_f(self, focal_len):
        self._focal_len = focal_len
        self._ratio = min(self._shape[0], self._shape[1]) / (self._focal_len * pi)

    def rand_f(self, f_range=[200, 400]):
        temp = random.random()
        self._focal_len = f_range[0] * (1 - temp) + f_range[1] * temp
        self._ratio = min(self._shape[0], self._shape[1]) / (self._focal_len * pi)

    def _init_pin_matrix(self, src_shape):
        rows = src_shape[0]
        cols = src_shape[1]
        self._pin_matrix = \
            np.array([
                [self._PARAM, 0, cols / 2, 0],
                [0, self._PARAM, rows / 2, 0],
                [0, 0, 1, 0]
            ])

    def print_ext_param(self):
        print("alpha:", self._alpha * 180 / pi, "度")
        print("beta:", self._beta * 180 / pi, "度")
        print("theta:", self._theta * 180 / pi, "度")

        print("X轴平移量：", self._x_trans)
        print("Y轴平移量：", self._y_trans)
        print("Z轴平移量：", self._z_trans)

    def set_ext_params(self, extParam):
        self._alpha = extParam[0] * pi / 180
        self._beta = extParam[1] * pi / 180
        self._theta = extParam[2] * pi / 180

        self._x_trans = extParam[3] * self._shape[1]
        self._y_trans = extParam[4] * self._shape[0]
        self._z_trans = extParam[5] * self._PARAM

    def rand_ext_params(self):
        temp = random.random()
        self._alpha = self.ALPHA_RANGE[0] * (1 - temp) + self.ALPHA_RANGE[1] * temp
        temp = random.random()
        self._beta = self.BETA_RANGE[0] * (1 - temp) + self.BETA_RANGE[1] * temp
        temp = random.random()
        self._theta = self.THETA_RANGE[0] * (1 - temp) + self.THETA_RANGE[1] * temp

        temp = random.random()
        self._x_trans = self.XTRANS_RANGE[0] * (1 - temp) + self.XTRANS_RANGE[1] * temp
        temp = random.random()
        self._y_trans = self.YTRANS_RANGE[0] * (1 - temp) + self.YTRANS_RANGE[1] * temp
        temp = random.random()
        self._z_trans = self.ZTRANS_RANGE[0] * (1 - temp) + self.ZTRANS_RANGE[1] * temp

    def _calc_cord_map(self, cv_img):
        self._init_ext_matrix()
        self._init_pin_matrix(cv_img.shape)

        src_rows = cv_img.shape[0]
        src_cols = cv_img.shape[1]
        
        dst_rows = self._shape[0]
        dst_cols = self._shape[1]

        cord_x, cord_y = np.meshgrid(np.arange(dst_cols), np.arange(dst_rows))
        cord = np.dstack((cord_x, cord_y)).astype(np.float64) - np.array([dst_cols / 2, dst_rows / 2])
        cord = cord.reshape(-1, 2)

        cord = np.array(cord) / self._ratio

        radius_array = np.sqrt(np.square(cord[:, 0]) + np.square(cord[:, 1]))
        theta_array = radius_array / self._focal_len
        if np.any(radius_array == 0):
            radius_array[radius_array == 0] = np.nan
        new_x_array = np.tan(theta_array) * cord[:, 0] / radius_array * self._focal_len
        new_y_array = np.tan(theta_array) * cord[:, 1] / radius_array * self._focal_len

        temp_index1 = radius_array == 0
        temp_index2 = cord[:, 0] == 0
        temp_index3 = cord[:, 1] == 0
        bad_x_index = temp_index1 | (temp_index2 & temp_index1)
        bad_y_index = temp_index1 | (temp_index3 & temp_index1)

        new_x_array[bad_x_index] = 0
        new_y_array[bad_y_index] = 0

        new_x_array = new_x_array.reshape((-1, 1))
        new_y_array = new_y_array.reshape((-1, 1))

        new_cord = np.hstack((new_x_array, new_y_array))
        new_cord = np.hstack((new_cord, np.ones((dst_rows * dst_cols, 1)) * self._PARAM))
        new_cord = np.hstack((new_cord, np.ones((dst_rows * dst_cols, 1))))

        pin_camera_array = np.matmul(self._rotate_trans_matrix, new_cord.T).T

        pin_image_cords = np.matmul(self._pin_matrix, pin_camera_array.T).T

        self._map_cols = pin_image_cords[:, 0] / pin_image_cords[:, 2]
        self._map_rows = pin_image_cords[:, 1] / pin_image_cords[:, 2]

        self._map_cols = self._map_cols.round().astype(int)
        self._map_rows = self._map_rows.round().astype(int)

        index1 = self._map_rows < 0
        index2 = self._map_rows >= src_rows
        index3 = self._map_cols < 0
        index4 = self._map_cols >= src_cols
        index5 = pin_image_cords[:, 2] <= 0

        bad_index = index1 | index2 | index3 | index4 | index5
        bad_index = bad_index | self._bad_index
        self._map_cols[bad_index] = cv_img.shape[1]
        self._map_rows[bad_index] = 0

    def _extend_img_color(self, cv_img):
        dst_img = np.hstack((cv_img, np.zeros((cv_img.shape[0], 1, 3), dtype=np.uint8)))
        dst_img[0, cv_img.shape[1]] = self._bkg_color
        return dst_img

    def _extend_img_gray(self, cv_img):
        dst_img = np.hstack((cv_img, np.zeros((cv_img.shape[0], 1), dtype=np.uint8)))
        dst_img[0, cv_img.shape[1]] = self._bkg_label
        return dst_img

    def transFromColor(self, cv_img, reuse=False):
        if not reuse:
            self._calc_cord_map(cv_img)

        cv_img = self._extend_img_color(cv_img)
        dst = np.array(cv_img[(self._map_rows, self._map_cols)])
        dst = dst.reshape(self._shape[0], self._shape[1], 3)
        return dst

    def transFromGray(self, cv_img, reuse=False):
        if not reuse:
            self._calc_cord_map(cv_img)

        cv_img = self._extend_img_gray(cv_img)
        dst = np.array(cv_img[(self._map_rows, self._map_cols)])
        dst = dst.reshape(self._shape[0], self._shape[1])
        return dst


def test_color():
    trans = FishEyeGenerator(250, [640, 640])
    img = cv2.imread("")
    trans.set_ext_param_range([25, 25, 25, 0.2, 0.2, 0.5])
    trans.rand_ext_params()
    trans.print_ext_param()
    s = time.time()
    dst = trans.transFromColor(img)
    e = time.time()
    print(e - s)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Source Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    plt.title("Transformed Image")
    plt.axis('off')

    plt.show()


def test_gray():
    trans = FishEyeGenerator(200, [640, 640])
    img = cv2.imread("", 0)
    dst = trans.transFromGray(img)
    dst *= 10
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    img *= 10

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Source Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    plt.title("Transformed Image")
    plt.axis('off')

    plt.show()


def process_folder(image_dir, annot_dir, output_image_dir, output_annot_dir):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_annot_dir, exist_ok=True)
    index = 0

    trans = FishEyeGenerator(250, [640, 640])
    
    trans.set_ext_param_range([25, 25, 25, 0.2, 0.2, 0.5])
    
    for filename in os.listdir(image_dir):
        index += 1
        if index % 25 == 0:
            print(index)
        if not filename.endswith('_leftImg8bit.png'):
            continue

        image_path = os.path.join(image_dir, filename)

        base_filename = filename.replace('_leftImg8bit.png', '')
        annot_filename = f"{base_filename}_gtFine_labelTrainIds.png"
        annot_path = os.path.join(annot_dir, annot_filename)

        if not os.path.exists(annot_path):
            print(f"Annotation file {annot_filename} not found, skipping.")
            continue

        img = cv2.imread(image_path)
        annot = cv2.imread(annot_path, 0)

        if img is None or annot is None:
            print(f"Failed to read {filename} or its annotation, skipping.")
            continue
        
        trans.rand_f()
        trans.rand_ext_params()

        transformed_img = trans.transFromColor(img)
        transformed_annot = trans.transFromGray(annot)

        output_image_path = os.path.join(output_image_dir, filename)
        output_annot_path = os.path.join(output_annot_dir, annot_filename)

        cv2.imwrite(output_image_path, transformed_img)
        cv2.imwrite(output_annot_path, transformed_annot)

    print("Finished！")


if __name__ == '__main__':
    image_dir = ""
    annot_dir = ""
    output_image_dir = ""
    output_annot_dir = ""

    process_folder(image_dir, annot_dir, output_image_dir, output_annot_dir)
