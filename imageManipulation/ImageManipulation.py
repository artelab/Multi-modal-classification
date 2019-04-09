import cv2


class ImageManipulator(object):

    def __init__(self, output_width):
        self.output_width = output_width

    def preprocess_images(self, bunch_of_images):
        images_list = []
        path_list = [image.decode('UTF-8') for image in bunch_of_images]
        for path in path_list:
            img = cv2.imread(path)
            img = cv2.resize(img, (self.output_width, self.output_width))
            img = img / 255
            images_list.append(img)
        return images_list
