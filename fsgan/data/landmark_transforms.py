import collections
import random
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from fsgan.utils.bbox_utils import scale_bbox, crop_img, hflip_bbox
from fsgan.utils.landmark_utils import generate_heatmaps, hflip_face_landmarks, align_crop


class LandmarksTransform(object):
    def __call__(self, img, landmarks, bbox):
        """
        Args:
            img (PIL Image or numpy.ndarray): Image to transform.
            landmarks (numpy.ndarray): Array of face landmarks (68 X 2)
            bbox (numpy.ndarray): Face bounding box (4,)

        Returns:
            Tensor: Converted image, landmarks, and bounding box.
        """
        return img, landmarks, bbox


class LandmarksPairTransform(object):
    def __call__(self, img1, landmarks1, bbox1, img2, landmarks2, bbox2):
        """
        Args:
            img1 (PIL Image or numpy.ndarray): First image to transform.
            landmarks1 (numpy.ndarray): First face landmarks (68 X 2)
            bbox1 (numpy.ndarray): First face bounding box (4,)
            img2 (PIL Image or numpy.ndarray): Second image to transform.
            landmarks2 (numpy.ndarray): Second face landmarks (68 X 2)
            bbox2 (numpy.ndarray): Second face bounding box (4,)

        Returns:
            Tensor: Converted image, landmarks, and bounding box.
        """
        return img1, landmarks1, bbox1, img2, landmarks2, bbox2


class Compose(LandmarksTransform):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> landmark_transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, landmarks=None, bboxes=None):
        for t in self.transforms:
            if isinstance(t, LandmarksTransform):
                img, landmarks, bboxes = t(img, landmarks, bboxes)
            else:
                img = t(img)

        return img, landmarks, bboxes

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ComposePair(LandmarksPairTransform):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> landmark_transforms.ComposePair([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms
        for t in self.transforms:
            assert isinstance(t, LandmarksPairTransform)

    def __call__(self,  img1, landmarks1, bbox1, img2, landmarks2, bbox2):
        for t in self.transforms:
            img1, landmarks1, bbox1, img2, landmarks2, bbox2 = t(img1, landmarks1, bbox1, img2, landmarks2, bbox2)

        return img1, landmarks1, bbox1, img2, landmarks2, bbox2

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(LandmarksTransform):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` with landmarks to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] with landmarks in numpy.ndarray format (68 x 2) to a torch.FloatTensor of shape
    ((C + 68) x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, img, landmarks, bbox):
        """
        Args:
            img (PIL Image or numpy.ndarray): Image to transform.
            landmarks (numpy.ndarray): Array of face landmarks (68 X 2)
            bbox (numpy.ndarray): Face bounding box (4,)

        Returns:
            Tensor: Converted image, landmarks, and bounding box.
        """
        img = F.to_tensor(img)
        landmarks = torch.from_numpy(landmarks)
        bbox = torch.from_numpy(bbox)
        return img, landmarks, bbox

    def __repr__(self):
        return self.__class__.__name__ + '()'
class ToTensor_segref(LandmarksTransform):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` with landmarks to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] with landmarks in numpy.ndarray format (68 x 2) to a torch.FloatTensor of shape
    ((C + 68) x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, img, landmarks, img_seg,imgref,landmarksref):
        """
        Args:
            img (PIL Image or numpy.ndarray): Image to transform.
            landmarks (numpy.ndarray): Array of face landmarks (68 X 2)
            bbox (numpy.ndarray): Face bounding box (4,)

        Returns:
            Tensor: Converted image, landmarks, and bounding box.
        """
        img = F.to_tensor(img)
        landmarks = torch.from_numpy(landmarks)
        img_seg = F.to_tensor(img_seg)
        for i in range(len(imgref)):
            imgref[i]=F.to_tensor(imgref[i])
        for i in range(len(landmarksref)):
            landmarksref[i]=torch.from_numpy(landmarksref[i])
        return img, landmarks, img_seg,imgref,landmarksref

    def __repr__(self):
        return self.__class__.__name__ + '()'
class ToTensor_seg(LandmarksTransform):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` with landmarks to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] with landmarks in numpy.ndarray format (68 x 2) to a torch.FloatTensor of shape
    ((C + 68) x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, img, landmarks, img_seg):
        """
        Args:
            img (PIL Image or numpy.ndarray): Image to transform.
            landmarks (numpy.ndarray): Array of face landmarks (68 X 2)
            bbox (numpy.ndarray): Face bounding box (4,)

        Returns:
            Tensor: Converted image, landmarks, and bounding box.
        """
        img = F.to_tensor(img)
        landmarks = torch.from_numpy(landmarks)
        img_seg = F.to_tensor(img_seg)
        return img, landmarks, img_seg

    def __repr__(self):
        return self.__class__.__name__ + '()'

class Resize(LandmarksTransform):
    """Resize the input PIL Image and it's corresponding landmarks to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, landmarks, bbox):
        """
        Args:
            img (PIL Image or numpy.ndarray): Image to resize.
            landmarks (numpy.ndarray): Array of face landmarks (68 X 2) or (68 X 3) to resize.
            bbox (numpy.ndarray): Face bounding box (4,)

        Returns:
            Tensor: Converted image, landmarks, and bounding box.
        """
        orig_size = np.array(img.size)
        img = F.resize(img, self.size, self.interpolation)
        axes_scale = (np.array(img.size) / orig_size)

        # 3D landmarks case
        if landmarks.shape[1] == 3:
            axes_scale = np.append(axes_scale, axes_scale.mean())

        landmarks *= axes_scale
        return img, landmarks, bbox

    def __repr__(self):
        interpolate_str = transforms._pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

class Resize_segref(LandmarksTransform):
    """Resize the input PIL Image and it's corresponding landmarks to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, landmarks, img_seg,imgref,landmarksref):
        """
        Args:
            img (PIL Image or numpy.ndarray): Image to resize.
            landmarks (numpy.ndarray): Array of face landmarks (68 X 2) or (68 X 3) to resize.
            bbox (numpy.ndarray): Face bounding box (4,)

        Returns:
            Tensor: Converted image, landmarks, and bounding box.
        """
        orig_size = np.array(img.size)
        img = F.resize(img, self.size, self.interpolation)
        axes_scale = (np.array(img.size) / orig_size)
        img_seg=F.resize(img_seg,self.size,self.interpolation)

        # 3D landmarks case
        if landmarks.shape[1] == 3:
            axes_scale = np.append(axes_scale, axes_scale.mean())

        landmarks *= axes_scale
        
        for i in range(len(imgref)):
            
            imgref[i]=F.resize(imgref[i],self.size,self.interpolation)
        for i in range(len(landmarksref)):
            landmarksref[i]*=axes_scale


        return img, landmarks, img_seg,imgref,landmarksref

    def __repr__(self):
        interpolate_str = transforms._pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class Resize_seg(LandmarksTransform):
    """Resize the input PIL Image and it's corresponding landmarks to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, landmarks, img_seg):
        """
        Args:
            img (PIL Image or numpy.ndarray): Image to resize.
            landmarks (numpy.ndarray): Array of face landmarks (68 X 2) or (68 X 3) to resize.
            bbox (numpy.ndarray): Face bounding box (4,)

        Returns:
            Tensor: Converted image, landmarks, and bounding box.
        """
        orig_size = np.array(img.size)
        img = F.resize(img, self.size, self.interpolation)
        axes_scale = (np.array(img.size) / orig_size)
        img_seg=F.resize(img_seg,self.size,self.interpolation)

        # 3D landmarks case
        if landmarks.shape[1] == 3:
            axes_scale = np.append(axes_scale, axes_scale.mean())

        landmarks *= axes_scale

        return img, landmarks, img_seg

    def __repr__(self):
        interpolate_str = transforms._pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)



class LandmarksToHeatmaps(LandmarksTransform):
    """Convert a ``numpy.ndarray`` landmarks vector to heatmaps.

    Converts a numpy.ndarray landmarks vector of (68 X 2) to heatmaps (H x W x 68) in the range [0, 255].
    """
    def __init__(self, sigma=None):
        self.sigma = sigma

    def __call__(self, img, landmarks, bbox):
        """
        Args:
            img (PIL Image or numpy.ndarray): Image to transform.
            landmarks (numpy.ndarray): Array of face landmarks (68 X 2)
            bbox (numpy.ndarray): Face bounding box (4,)

        Returns:
            Tensor: Converted image.
        """
        landmarks = generate_heatmaps(img.size[1], img.size[0], landmarks, sigma=self.sigma)
        return img, landmarks, bbox

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={0})'.format(self.sigma)

class LandmarksToHeatmaps_segref(LandmarksTransform):
    """Convert a ``numpy.ndarray`` landmarks vector to heatmaps.

    Converts a numpy.ndarray landmarks vector of (68 X 2) to heatmaps (H x W x 68) in the range [0, 255].
    """
    def __init__(self, sigma=None):
        self.sigma = sigma

    def __call__(self, img, landmarks, img_seg,imgref,landmarksref):
        """
        Args:
            img (PIL Image or numpy.ndarray): Image to transform.
            landmarks (numpy.ndarray): Array of face landmarks (68 X 2)
            bbox (numpy.ndarray): Face bounding box (4,)

        Returns:
            Tensor: Converted image.
        """
        landmarks = generate_heatmaps(img.size[1], img.size[0], landmarks, sigma=self.sigma)
        for i in range(len(landmarksref)):
            landmarksref[i]=generate_heatmaps(img.size[1],img.size[0],landmarksref[i],sigma=self.sigma)
        return img, landmarks, img_seg,imgref,landmarksref

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={0})'.format(self.sigma)


class LandmarksToHeatmaps_seg(LandmarksTransform):
    """Convert a ``numpy.ndarray`` landmarks vector to heatmaps.

    Converts a numpy.ndarray landmarks vector of (68 X 2) to heatmaps (H x W x 68) in the range [0, 255].
    """
    def __init__(self, sigma=None):
        self.sigma = sigma

    def __call__(self, img, landmarks, img_seg):
        """
        Args:
            img (PIL Image or numpy.ndarray): Image to transform.
            landmarks (numpy.ndarray): Array of face landmarks (68 X 2)
            bbox (numpy.ndarray): Face bounding box (4,)

        Returns:
            Tensor: Converted image.
        """
        landmarks = generate_heatmaps(img.size[1], img.size[0], landmarks, sigma=self.sigma)
        return img, landmarks, img_seg

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={0})'.format(self.sigma)


class FaceCropsegref(LandmarksTransform):
    """Aligns and crops pil face images.

    Args:
        bbox_scale (float): Multiplier factor to scale tight bounding box
        bbox_square (bool): Force crop to be square.
        align (bool): Toggle face alignment using landmarks.
    """

    def __init__(self, height,width, align=False):
        self.height = height
        self.width = width
        self.align = align

    def __call__(self, img, landmarks, img_seg,imgref,landmarksref):
        """
        Args:
            img (PIL Image): Face image to align and crop.
            landmarks (numpy array): Face landmarks
            bbox (numpy array): Face tight bounding box

        Returns:
            PIL Image: Rescaled image.
        """
        #img = np.array(img).copy()
        #img_seg=np.array(img_seg).copy()
        left=(img.size[0]-self.width)//2
        upper=(img.size[1]-self.height)//2
        right=left+self.width
        lower=upper+self.height
        img=img.crop((left,upper,right,lower))
        img_seg=img_seg.crop((left,upper,right,lower))
        landmarks=landmarks-[left,upper]
        for i in range(len(imgref)):
            imgref[i]=imgref[i].crop((left,upper,right,lower))
        for i in range(len(landmarksref)):
            landmarksref[i]=landmarksref[i]-[left,upper]

        return img, landmarks, img_seg,imgref,landmarksref

    def __repr__(self):
        return self.__class__.__name__ + '(bbox_scale={0}, bbox_square={1}, align={2})'.format(
            self.bbox_scale, self.bbox_square, self.align)


class FaceCropseg(LandmarksTransform):
    """Aligns and crops pil face images.

    Args:
        bbox_scale (float): Multiplier factor to scale tight bounding box
        bbox_square (bool): Force crop to be square.
        align (bool): Toggle face alignment using landmarks.
    """

    def __init__(self, height,width, align=False):
        self.height = height
        self.width = width
        self.align = align

    def __call__(self, img, landmarks, img_seg):
        """
        Args:
            img (PIL Image): Face image to align and crop.
            landmarks (numpy array): Face landmarks
            bbox (numpy array): Face tight bounding box

        Returns:
            PIL Image: Rescaled image.
        """
        #img = np.array(img).copy()
        #img_seg=np.array(img_seg).copy()
        left=(img.size[0]-self.width)//2
        upper=(img.size[1]-self.height)//2
        right=left+self.width
        lower=upper+self.height
        img=img.crop((left,upper,right,lower))
        img_seg=img_seg.crop((left,upper,right,lower))
        landmarks=landmarks-[left,upper]

        return img, landmarks, img_seg

    def __repr__(self):
        return self.__class__.__name__ + '(bbox_scale={0}, bbox_square={1}, align={2})'.format(
            self.bbox_scale, self.bbox_square, self.align)


class FaceAlignCrop(LandmarksTransform):
    """Aligns and crops pil face images.

    Args:
        bbox_scale (float): Multiplier factor to scale tight bounding box
        bbox_square (bool): Force crop to be square.
        align (bool): Toggle face alignment using landmarks.
    """

    def __init__(self, bbox_scale=2.0, bbox_square=True, align=False):
        self.bbox_scale = bbox_scale
        self.bbox_square = bbox_square
        self.align = align

    def __call__(self, img, landmarks, bbox):
        """
        Args:
            img (PIL Image): Face image to align and crop.
            landmarks (numpy array): Face landmarks
            bbox (numpy array): Face tight bounding box

        Returns:
            PIL Image: Rescaled image.
        """
        img = np.array(img).copy()
        if self.align:
            img, landmarks = align_crop(img, landmarks, bbox, self.bbox_scale, self.bbox_square)
        else:
            bbox_scaled = scale_bbox(bbox, self.bbox_scale, self.bbox_square)
            img, landmarks = crop_img(img, landmarks, bbox_scaled)

        img = Image.fromarray(img)

        return img, landmarks, bbox

    def __repr__(self):
        return self.__class__.__name__ + '(bbox_scale={0}, bbox_square={1}, align={2})'.format(
            self.bbox_scale, self.bbox_square, self.align)


class RandomHorizontalFlipPair(LandmarksPairTransform):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, landmarks1, bbox1, img2, landmarks2, bbox2):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            img1 = F.hflip(img1)
            img2 = F.hflip(img2)
            landmarks1 = hflip_face_landmarks(landmarks1, img1.size[0])
            landmarks2 = hflip_face_landmarks(landmarks2, img2.size[0])
            bbox1 = hflip_bbox(bbox1, img1.size[0])
            bbox2 = hflip_bbox(bbox2, img2.size[0])

        return img1, landmarks1, bbox1, img2, landmarks2, bbox2

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class Pyramids_segref(LandmarksTransform):
    """Generate pyramids from an image and it's corresponding landmarks and bounding box

    Args:
        levels (int): number of pyramid levels (must be 1 or greater)
    """
    def __init__(self, levels=1):
        assert levels >= 1
        self.levels = levels

    def __call__(self, img, landmarks, img_seg,imgref,landmarksref):
        """
        Args:
            img (PIL Image): Face image to align and crop.
            landmarks (numpy array): Face landmarks
            bbox (numpy array): Face tight bounding box

        Returns:
            Image list (PIL Image list): List of scaled images.
            Landmarks list (numpy array list): List of scaled landmarks.
            Bounding boxes list (numpy array list): List of scaled boudning boxes.
        """
        img_pyd = [img]
        landmarks_pyd = [landmarks]
        img_seg_pyd = [img_seg]
        for i in range(self.levels - 1):
            img_pyd.append(Image.fromarray(cv2.pyrDown(np.array(img_pyd[-1]))))
            landmarks_pyd.append(landmarks_pyd[-1] / 2)
            img_seg_pyd.append(Image.fromarray(cv2.pyrDown(np.array(img_seg_pyd[-1]))))
        
        imgref=[imgref]
        landmarksref=[landmarksref]
        for i in range(self.levels-1):
            tmpimg=[]
            tmplandmark=[]
            for j in range(len(imgref[0])):
                tmpimg.append(Image.fromarray(cv2.pyrDown(np.array(imgref[-1][j]))))
            for j in range(len(landmarksref[0])):
                tmplandmark.append(landmarksref[-1][j] / 2)
            imgref.append(tmpimg)
            landmarksref.append(tmplandmark)

        


        return img_pyd, landmarks_pyd, img_seg_pyd,imgref,landmarksref

    def __repr__(self):
        return self.__class__.__name__ + '(levels={})'.format(self.levels)

class Pyramids_seg(LandmarksTransform):
    """Generate pyramids from an image and it's corresponding landmarks and bounding box

    Args:
        levels (int): number of pyramid levels (must be 1 or greater)
    """
    def __init__(self, levels=1):
        assert levels >= 1
        self.levels = levels

    def __call__(self, img, landmarks, img_seg):
        """
        Args:
            img (PIL Image): Face image to align and crop.
            landmarks (numpy array): Face landmarks
            bbox (numpy array): Face tight bounding box

        Returns:
            Image list (PIL Image list): List of scaled images.
            Landmarks list (numpy array list): List of scaled landmarks.
            Bounding boxes list (numpy array list): List of scaled boudning boxes.
        """
        img_pyd = [img]
        landmarks_pyd = [landmarks]
        img_seg_pyd = [img_seg]
        for i in range(self.levels - 1):
            img_pyd.append(Image.fromarray(cv2.pyrDown(np.array(img_pyd[-1]))))
            landmarks_pyd.append(landmarks_pyd[-1] / 2)
            img_seg_pyd.append(Image.fromarray(cv2.pyrDown(np.array(img_seg_pyd[-1]))))

        return img_pyd, landmarks_pyd, img_seg_pyd

    def __repr__(self):
        return self.__class__.__name__ + '(levels={})'.format(self.levels)


class Pyramids(LandmarksTransform):
    """Generate pyramids from an image and it's corresponding landmarks and bounding box

    Args:
        levels (int): number of pyramid levels (must be 1 or greater)
    """
    def __init__(self, levels=1):
        assert levels >= 1
        self.levels = levels

    def __call__(self, img, landmarks, bbox):
        """
        Args:
            img (PIL Image): Face image to align and crop.
            landmarks (numpy array): Face landmarks
            bbox (numpy array): Face tight bounding box

        Returns:
            Image list (PIL Image list): List of scaled images.
            Landmarks list (numpy array list): List of scaled landmarks.
            Bounding boxes list (numpy array list): List of scaled boudning boxes.
        """
        img_pyd = [img]
        landmarks_pyd = [landmarks]
        bbox_pyd = [bbox]
        for i in range(self.levels - 1):
            img_pyd.append(Image.fromarray(cv2.pyrDown(np.array(img_pyd[-1]))))
            landmarks_pyd.append(landmarks_pyd[-1] / 2)
            bbox_pyd.append(bbox_pyd[-1] / 2)

        return img_pyd, landmarks_pyd, bbox_pyd

    def __repr__(self):
        return self.__class__.__name__ + '(levels={})'.format(self.levels)


class ComposePyramidssegref(LandmarksTransform):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> landmark_transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, landmarks=None, img_seg=None,imgref=None, landmarksref=None):
        for t in self.transforms:
            if isinstance(t, LandmarksTransform):
                if isinstance(img, list):
                    for i in range(len(img)):
                        img[i], landmarks[i], img_seg[i],imgref[i],landmarksref[i] = t(img[i], landmarks[i], img_seg[i],imgref[i],landmarksref[i])
                else:
                    img, landmarks, img_seg,imgref,landmarksref = t(img, landmarks, img_seg,imgref,landmarksref)
            else:
                if isinstance(img, list):
                    for i in range(len(img)):
                        img[i] = t(img[i])
                        for j in range(len(imgref[i])):
                            imgref[i][j]=t(imgref[i][j])
                else:
                    img = t(img)
                    for j in range(len(imgref)):
                        imgref[j]=t(imgref[j])

        return img, landmarks, img_seg, imgref,landmarksref

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ComposePyramids(LandmarksTransform):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> landmark_transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, landmarks=None, bboxes=None):
        for t in self.transforms:
            if isinstance(t, LandmarksTransform):
                if isinstance(img, list):
                    for i in range(len(img)):
                        img[i], landmarks[i], bboxes[i] = t(img[i], landmarks[i], bboxes[i])
                else:
                    img, landmarks, bboxes = t(img, landmarks, bboxes)
            else:
                if isinstance(img, list):
                    for i in range(len(img)):
                        img[i] = t(img[i])
                else:
                    img = t(img)

        return img, landmarks, bboxes

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string