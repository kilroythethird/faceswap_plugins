#!/usr/bin/env python3
""" S3FD Face detection plugin
https://arxiv.org/abs/1708.05237

Adapted from S3FD Port in FAN:
https://github.com/1adrianb/face-alignment
"""

from scipy.special import logsumexp

import numpy as np

from lib.multithreading import MultiThread
from ._base import Detector, logger
import cv2


class Detect(Detector):
    """ S3FD detector for face recognition """
    def __init__(self, **kwargs):
        git_model_id = 3
        model_filename = "s3fd_v1.pb"
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)
        self.name = "s3fd"
        self.target = (640, 640)  # Uses approx 4 GB of VRAM
        self.vram = 4096
        self.min_vram = 1024  # Will run at this with warnings
        self.model = None
        self.supports_plaidml = True

    def initialize(self, *args, **kwargs):
        """ Create the s3fd detector """
        try:
            super().initialize(*args, **kwargs)
            logger.info("Initializing S3FD Detector...")
            card_id, vram_free, vram_total = self.get_vram_free()
            if vram_free <= self.vram:
                tf_ratio = 1.0
            else:
                tf_ratio = self.vram / vram_total

            logger.verbose("Reserving %s%% of total VRAM per s3fd thread",
                           round(tf_ratio * 100, 2))

            confidence = self.config["confidence"] / 100
            self.model = S3fd(self.model_path, self.target, tf_ratio, card_id, confidence)
            logger.verbose("Processing in %s threads", self.batch_size)

            self.init.set()
            logger.info("Initialized S3FD Detector.")
        except Exception as err:
            self.error.set()
            raise err

    def detect_faces(self, *args, **kwargs):
        """ Detect faces in Multiple Threads """
        super().detect_faces(*args, **kwargs)
        workers = MultiThread(target=self.detect_thread, thread_count=self.batch_size)
        workers.start()
        workers.join()
        sentinel = self.queues["in"].get()
        self.queues["out"].put(sentinel)
        logger.debug("Detecting Faces complete")

    def detect_thread(self):
        """ Detect faces in rgb image """
        logger.debug("Launching Detect")
        while True:
            item = self.get_item()
            if item == "EOF":
                break
            logger.trace("Detecting faces: '%s'", item["filename"])

            if max(item["image"].shape[:2]) > self.target[0]:
                detect_image, scale = self.compile_detection_image(
                    item["image"], is_square=True, pad_to=self.target
                )
            else:
                detect_image = self.pad_image(item["image"], self.target)
                scale = 1.0
            faces = self.model.detect_face(detect_image)

            detected_faces = self.process_output(faces, None, scale)
            item["detected_faces"] = detected_faces
            self.finalize(item)

        logger.debug("Thread Completed Detect")

    def process_output(self, faces, rotation_matrix, scale):
        """ Compile found faces for output """
        faces = [self.to_bounding_box_dict(face[0], face[1], face[2], face[3]) for face in faces]
        detected = [self.to_bounding_box_dict(face["left"] / scale, face["top"] / scale,
                                              face["right"] / scale, face["bottom"] / scale)
                    for face in faces]
        logger.trace("Processed Output: %s", detected)
        return detected

    def compile_detection_image(self, input_image,
                                is_square=False, scale_up=False, to_rgb=False,
                                pad_to=None, to_grayscale=False):
        """ Compile the detection image """
        image = input_image.copy()
        if to_rgb:
            image = image[:, :, ::-1]
        elif to_grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # pylint: disable=no-member
        scale = self.set_scale(image, is_square=is_square, scale_up=scale_up)
        image = self.scale_image(image, scale, pad_to)
        return [image, scale]

    @staticmethod
    def scale_image(image, scale, pad_to):
        """ Scale the image """
        # pylint: disable=no-member
        canvas_size = pad_to
        height, width = image.shape[:2]
        interpln = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
        if scale != 1.0:
            dims = (int(width * scale), int(height * scale))
            if scale < 1.0:
                logger.info("Resizing image from %sx%s to %s. Scale=%s",
                             width, height, "x".join(str(i) for i in canvas_size), scale)
            image = cv2.resize(image, dims, interpolation=interpln)
        else:
            dims = (width, height)
        if pad_to:
            image = Detect.pad_image(image, pad_to)
        return image

    @staticmethod
    def pad_image(image, target):
        height, width = image.shape[:2]
        if width < target[0] or height < target[1]:
            return cv2.copyMakeBorder(
                image, 0, target[1] - height, 0, target[0] - width,
                cv2.BORDER_CONSTANT, (0,0,0)
            )
        return image


import keras
import os
class S3fd():
    """ Tensorflow Network """
    def __init__(self, model_path, target_size, vram_ratio, card_id, confidence):
        logger.debug("Initializing: %s: (model_path: '%s', target_size: %s, vram_ratio: %s, "
                     "card_id: %s)",
                     self.__class__.__name__, model_path, target_size, vram_ratio, card_id)
        url = "https://share.gnutp.com/stuff/S3FD.h5"
        cache_path = os.path.dirname(__file__)
        cache_path = keras.utils.get_file("S3FD.h5", url, cache_dir=cache_path, cache_subdir=".cache")
        self.model_path = cache_path
        self.confidence = confidence
        self.graph = self.load_graph()
        logger.debug("Initialized: %s", self.__class__.__name__)

    def load_graph(self):
        """ Load the tensorflow Model and weights """
        # pylint: disable=not-context-manager
        logger.verbose("Initializing S3FD Network model...")
        model = keras.models.load_model(self.model_path)
        #model.summary(line_length=189)
        return model

    def detect_face(self, feed_item):
        """ Detect faces """
        #logger.info("Got image with shape %s", str(feed_item.shape))
        feed_item = feed_item - np.array([104.0, 117.0, 123.0])
        #feed_item = feed_item.transpose(2, 0, 1)
        feed_item = feed_item.reshape((1,) + feed_item.shape).astype('float32')
        #bboxlist = self.session.run(self.output, feed_dict={self.input: feed_item})
        bboxlist = self.graph.predict(feed_item) #session.run(self.output, feed_dict={self.input: feed_item})
        #logger.info("Predicted image with shape %s (%s)", len(bboxlist), str(bboxlist[0].shape))
        bboxlist = self.post_process(bboxlist)

        keep = self.nms(bboxlist, 0.3)
        bboxlist = bboxlist[keep, :]
        bboxlist = [x for x in bboxlist if x[-1] >= self.confidence]

        return np.array(bboxlist)

    def post_process(self, bboxlist):
        """ Perform post processing on output """
        retval = list()
        for i, ((ocls,), (oreg,)) in enumerate ( zip ( bboxlist[::2], bboxlist[1::2] ) ):
            stride = 2**(i + 2)    # 4,8,16,32,64,128
            s_d2 = stride / 2
            s_m4 = stride * 4

            for hindex, windex in zip(*np.where(ocls > 0.05)):
                score = ocls[hindex, windex]
                loc   = oreg[hindex, windex, :]
                priors = np.array([windex * stride + s_d2, hindex * stride + s_d2, s_m4, s_m4])
                priors_2p = priors[2:]
                box = np.concatenate((priors[:2] + loc[:2] * 0.1 * priors_2p,
                                      priors_2p * np.exp(loc[2:] * 0.2)) )
                box[:2] -= box[2:] / 2
                box[2:] += box[:2]
                retval.append([*box, score])
        retval = np.array(retval)
        if len(retval) == 0:
            retval = np.zeros((1, 5))
        return retval

    @staticmethod
    def softmax(inp, axis):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(inp - logsumexp(inp, axis=axis, keepdims=True))

    @staticmethod
    def decode(loc, priors, variances):
        """Decode locations from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            loc (tensor): location predictions for loc layers,
                Shape: [num_priors,4]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded bounding box predictions
        """
        boxes = np.concatenate((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                                priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])),
                               1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    @staticmethod
    def nms(dets, thresh):
        """ Perform Non-Maximum Suppression """
        keep = list()
        if len(dets) == 0:
            return keep

        x_1, y_1, x_2, y_2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x_2 - x_1 + 1) * (y_2 - y_1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx_1, yy_1 = np.maximum(x_1[i], x_1[order[1:]]), np.maximum(y_1[i], y_1[order[1:]])
            xx_2, yy_2 = np.minimum(x_2[i], x_2[order[1:]]), np.minimum(y_2[i], y_2[order[1:]])

            width, height = np.maximum(0.0, xx_2 - xx_1 + 1), np.maximum(0.0, yy_2 - yy_1 + 1)
            ovr = width * height / (areas[i] + areas[order[1:]] - width * height)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep
