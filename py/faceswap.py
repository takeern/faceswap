import os
import dlib
import numpy
import cv2

predictor_path = os.path.join(os.path.dirname(__file__), "../", "shape_predictor_68_face_landmarks.dat")
jolie = os.path.join(os.path.dirname(__file__), "../", "source", "brad.jpg")
brad = os.path.join(os.path.dirname(__file__), "../", "source", "test1.jpeg")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

class TooManyFaces(Exception):
  pass

class NoFaces(Exception):
  pass

# 获取人脸关键点
def get_landmarks(im):
  rects = detector(im, 1)

  if len(rects) == 0:
    raise NoFaces
  return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def read_im_and_landmarks(fname):
  im = cv2.imread(fname, cv2.IMREAD_COLOR)
  im = cv2.resize(im, (im.shape[1] * 1, im.shape[0] * 1))
  s = get_landmarks(im)
  # print('mask: ', s);
  return im, s

def transformation_from_points(points1, points2):
  # print('输入 mask: ', points1)
  points1 = points1.astype(numpy.float64)
  points2 = points2.astype(numpy.float64)

  c1 = numpy.mean(points1, 0)
  c2 = numpy.mean(points2, 0)
  points1 -= c1
  points2 -= c2

  s1 = numpy.std(points1)
  s2 = numpy.std(points2)
  points1 /= s1
  points2 /= s2

  U, S, Vt = numpy.linalg.svd(points1.T * points2)
  R = (U * Vt).T

  return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                      c2.T - (s2 / s1) * R * c1.T)),
                        numpy.matrix([0., 0., 1.])])
#  对图像进行旋转
def warp_im(im, M, dshape):
  output_im = numpy.zeros(dshape, dtype=im.dtype)
  cv2.warpAffine(im, M[:2], (dshape[1], dshape[0]), dst=output_im, borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)
  return output_im

def draw_convex_hull(im, points, color):
  # 获取节点凸包 https://www.cnblogs.com/jclian91/p/9728488.html
  points = cv2.convexHull(points)
  # 填充凸包内颜色 https://blog.csdn.net/lyxleft/article/details/90676451
  cv2.fillConvexPoly(im, points, color=color)


def get_face_mask(im, marks):
  mask = numpy.zeros(im.shape[:2], dtype=numpy.float64)

  for group in OVERLAY_POINTS:
    # print(group)
    draw_convex_hull(mask, marks[group], color=1)

  # cv2.imshow('test1', mask)
  mask = numpy.array([mask, mask, mask]).transpose((1,2,0))

  # mask = (cv2.GaussianBlur(mask, (11, 11), 0) > 0) * 1.0
  # cv2.imshow('test2', mask)
  mask = cv2.GaussianBlur(mask, (11, 11), 0)
  # cv2.imshow('test3', mask)
  return mask

def correct_colors(im1, im2, marks):
  blur_amount = 0.6 * numpy.linalg.norm(
    numpy.mean(marks[LEFT_EYE_POINTS], axis=0) - numpy.mean(marks[RIGHT_EYE_POINTS], axis=0)
  )

  # print('blur_amount float', blur_amount)
  blur_amount = int(blur_amount)
  # print('blur_amount int', blur_amount)
  if blur_amount % 2 == 0:
    blur_amount +=1
  im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
  im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
  cv2.imshow('im1_blur', im1_blur)
  cv2.imshow('im2_blur', im2_blur)

  im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

  return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) / im2_blur.astype(numpy.float64))

im1, marks1 = read_im_and_landmarks(jolie)
im2, marks2 = read_im_and_landmarks(brad)
M = transformation_from_points(marks1, marks2)
# print('M,', M, M[:2])
mask1 = get_face_mask(im1, marks1)
mask2 = get_face_mask(im2, marks2)

rotate_mask2 = warp_im(mask2, M, im1.shape)
combine_mask = numpy.max([mask1, rotate_mask2], axis=0)
rotate_im2 = warp_im(im2, M, im1.shape)


warped_corrected_im2 = correct_colors(im1, rotate_im2, marks1)
cv2.imshow('warped_corrected_im2', warped_corrected_im2)

output_im = im1 * (1.0 - combine_mask) + warped_corrected_im2 * combine_mask

# print('combine_mask', combine_mask.shape)
# cv2.imshow('mask1', mask1)
# cv2.imshow('rotate_mask2', rotate_mask2)
# cv2.imshow('combine_mask', combine_mask)
# cv2.imshow('tets4', output_im)

cv2.imwrite('output1.jpg', output_im)
# cv2.imshow('rotate_im2', rotate_im2)

# print('OVERLAY_POINTS', marks1)

# cv2.imshow('test1', output_im)
# cv2.waitKey(0)
# cv2.destoryAllWindows()