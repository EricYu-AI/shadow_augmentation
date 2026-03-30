from typing import Tuple
import numpy as np
import cv2
import random

from scipy.special import binom
from skimage.util import random_noise
from skimage.transform import resize


class BezierCurve:
    def __init__(
        self,
        npts: int = 5,
        rad: float = 0.3,
        edgy: float = 0,
        scale: float = 0.8,
        numpoints: int = 100,
    ) -> None:
        """given an array of points *a*, create a curve through
        those points.

        Args:
            npts: the number of random points to use. Of course the minimum
                number of points is 3. The more points you use, the more
                feature rich the shapes can become; at the risk of creating
                overlaps or loops in the curve.
            rad: the radius around the points at which the control points of
                the bezier curve sit.This number is relative to the distance
                between adjacent points and should hence be between 0 and 1.
                The larger the radius, the sharper the features of the curve.
            edgy: a parameter to determine the smoothness of the curve. If 0
                the angle of the curve through each point will be the mean
                between the direction to adjacent points.
                The larger it gets, the more the angle will be determined
                only by one adjacent point. The curve hence gets "edgier".
                edgy=0 is smoothest.
            scale: initial random points scale w.r.t 1
            numpoints: number of points used for curve generation
        """
        self.npts = npts
        self.scale = scale
        self.numpoints = numpoints
        self.rad = rad
        self.edgy = edgy

    def calc_intermediate_points(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        angle1: float,
        angle2: float,
    ) -> None:
        d = np.sqrt(np.sum((p2 - p1) ** 2))
        _rad = self.rad * d
        _p = np.zeros((4, 2))
        _p[0, :] = p1[:]
        _p[3, :] = p2[:]
        _p[1, :] = p1 + np.array(
            [_rad * np.cos(angle1), _rad * np.sin(angle1)]
        )
        _p[2, :] = p2 + np.array(
            [_rad * np.cos(angle2 + np.pi), _rad * np.sin(angle2 + np.pi)]
        )
        return self.bezier(_p)

    def bezier(self, points: np.ndarray) -> np.ndarray:
        def bernstein(n: int, k: int, t: np.ndarray) -> np.ndarray:
            return binom(n, k) * t ** k * (1.0 - t) ** (n - k)

        N = len(points)
        t = np.linspace(0, 1, num=self.numpoints)
        curve = np.zeros((self.numpoints, 2))
        for i in range(N):
            curve += np.outer(bernstein(N - 1, i, t), points[i])
        return curve

    def get_curve(self, points: np.ndarray) -> np.ndarray:
        segments = []
        for i in range(len(points) - 1):
            seg = self.calc_intermediate_points(
                points[i, :2],
                points[i + 1, :2],
                points[i, 2],
                points[i + 1, 2],
            )
            segments.append(seg)
        curve = np.concatenate(segments)
        return curve

    def _ccw_sort(self, p: np.ndarray) -> np.ndarray:
        d = p - np.mean(p, axis=0)
        s = np.arctan2(d[:, 0], d[:, 1])
        return p[np.argsort(s), :]

    def get_random_points(
        self, min_dst: float = None, rec: int = 0, max_try: int = 200
    ) -> np.ndarray:
        """create n random points in the unit square, which are *min_dst*
        apart, then scale them.
        """
        min_dst = min_dst or 0.7 / self.npts
        a = np.random.rand(self.npts, 2)
        d = np.sqrt(np.sum(np.diff(self._ccw_sort(a), axis=0), axis=1) ** 2)
        if np.all(d >= min_dst) or rec >= max_try:
            return a * self.scale
        else:
            return self.get_random_points(min_dst=min_dst, rec=rec + 1)

    def __call__(self) -> np.ndarray:
        a = self.get_random_points()
        p = np.arctan(self.edgy) / np.pi + 0.5
        a = self._ccw_sort(a)
        a = np.append(a, np.atleast_2d(a[0, :]), axis=0)
        d = np.diff(a, axis=0)
        ang = np.arctan2(d[:, 1], d[:, 0])
        ang = (ang >= 0) * ang + (ang < 0) * (ang + 2 * np.pi)
        ang1 = ang
        ang2 = np.roll(ang, 1)
        ang = p * ang1 + (1 - p) * ang2 + (np.abs(ang2 - ang1) > np.pi) * np.pi
        ang = np.append(ang, [ang[0]])
        a = np.append(a, np.atleast_2d(ang).T, axis=1)
        return self.get_curve(a)


def noisy_image(shape: Tuple[int, int]) -> np.ndarray:

    imh, imw = shape[:2]
    val = random.uniform(0.036, 0.107)
    noisy1 = np.zeros((imh, imw))
    noisy1 = random_noise(noisy1, mode="gaussian", var=val ** 2, clip=False)

    # Half resolution
    noisy2 = np.zeros((imh // 2, imw // 2))
    noisy2 = random_noise(
        noisy2, mode="gaussian", var=(val * 2) ** 2, clip=False
    )  # Use val*2 (needs tuning...)
    noisy2 = resize(noisy2, (imh, imw))  # Upscale to original image size

    # Quarter resolution
    noisy3 = np.zeros((imh // 4, imw // 4))
    noisy3 = random_noise(
        noisy3, mode="gaussian", var=(val * 4) ** 2, clip=False
    )  # Use val*4 (needs tuning...)
    noisy3 = resize(noisy3, (imh, imw))  # What is the interpolation method?

    noisy = (noisy1 + noisy2 + noisy3 + 0.5).clip(0, 1)
    return noisy


class AugmentShadowOnFeature:
    def __init__(
        self,
        shadow_size: int = 400,
        imsize: int = 1280,
        blur: bool = True,
        hsigma: Tuple = (100, 110),
        ssigma: Tuple = (100, 130),
        vsigma: Tuple = (40, 50),
    ) -> None:
        self.shadow_size = shadow_size
        self.imsize = imsize
        self.blur = blur
        self.hsigma = hsigma
        self.ssigma = ssigma
        self.vsigma = vsigma
        self.get_bezier_curve = BezierCurve(
            rad=0.2, edgy=0.1, scale=0.85, numpoints=100
        )

    def apply_random_shadow(
        self, im: np.ndarray, linecoords: np.ndarray, demo: str = ""
    ) -> None:
        """Augment image with random shadow

        Assumptions:
            1. Square image
        Args:
            im: (1280, 1280, 3) uint8 255 0
            linecoords: (2, 2) int64
            demo: if given, display demo images for dev
        """
        if linecoords.shape == (2, 2):
            center_at = np.mean(linecoords, axis=0).astype(int)
        elif linecoords.shape == (2,):
            center_at = linecoords
        else:
            raise NotImplementedError
        coord = (center_at - self.shadow_size // 2).clip(0, self.imsize - 1)
        coord_br = (center_at + self.shadow_size // 2).clip(0, self.imsize - 1)
        crop = im[coord[1]:coord_br[1], coord[0]:coord_br[0]]
        if min(crop.shape) <= 0:
            return

        ch, cw = crop.shape[:2]
        crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        hueV = random.randrange(self.hsigma[0], self.hsigma[1])
        satV = random.randrange(self.ssigma[0], self.ssigma[1])
        valV = random.randrange(self.vsigma[0], self.vsigma[1])
        crop_hsv[..., 0] = np.ones_like(crop_hsv[..., 0]) * hueV
        crop_hsv[..., 1] = np.ones_like(crop_hsv[..., 1]) * satV
        valmean = np.mean(crop_hsv[..., -1])
        if valmean > 1:
            valfactor = valV / valmean
            noisy_level = random.uniform(0.1, 0.2)
            noise = noisy_image(crop.shape)  # float in range 0~1
            darkness = valfactor + noisy_level * noise
            crop_hsv[..., -1] = crop_hsv[..., -1] * darkness

        crop_dark = cv2.cvtColor(crop_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        if self.blur:
            ksize = random.randrange(1, 4)
            crop_dark = cv2.blur(crop_dark, (ksize, ksize))
        contour = self.get_bezier_curve() * np.array((cw, ch))
        contour = contour.astype(int)
        aug_mask = np.zeros((ch, cw), dtype=np.uint8)
        cv2.fillPoly(aug_mask, pts=[contour], color=255)
        # aug_mask = np.ones((ch, cw), dtype=np.uint8) * 255  # debug
        crop[aug_mask == 255] = crop_dark[aug_mask == 255]

        if demo:
            crop = cv2.resize(crop, (0, 0), fx=2, fy=2)
            crop_dark = cv2.resize(crop_dark, (0, 0), fx=2, fy=2)
            cv2.imshow("mask_" + demo, aug_mask)
            cv2.imshow("crop_" + demo, crop)
            cv2.imshow("shadowed_" + demo, crop_dark)


def main() -> None:

    from pathlib import Path
    from point_rend.tile_utils import (
        convert_lonlat_to_tile_xy,
        load_feature_from_json,
    )

    # shadow augmentation engine
    aug_shadow = AugmentShadowOnFeature(blur=True)

    # data
    home = Path("~").expanduser()
    data_path = home / "HERE/crosswalk"
    label_dir = data_path / "gco/test10"
    image_dir = data_path / "images/test10"

    for label_json in label_dir.glob("*.json"):
        tile_id = label_json.stem
        imf = image_dir / f"{tile_id}.png"

        try:
            im = cv2.imread(str(imf))
        except Exception as e:
            print(tile_id, " : ", e)

        geo_feature_lonlat, feat_cntr_lonlat = load_feature_from_json(
            label_json,
            features_oi=["LOGICAL_CROSS_WALKS", "STRIPES_CROSS_WALKS"],
        )
        # print(geo_feature_lonlat)
        if geo_feature_lonlat:
            centers_pxy = convert_lonlat_to_tile_xy(feat_cntr_lonlat, tile_id)
            for si, cntr in enumerate(centers_pxy):
                ftr = cntr.clip(0, 1279)
                # cv2.line(im, coord_1, coord_2, (0, 255, 0), 3)
                px, py = ftr
                cv2.circle(im, (px, py), 1, (0, 0, 255), 3, 8, 0)

                if random.uniform(0, 1) > 0.59:
                    continue
                aug_shadow.apply_random_shadow(im, ftr, demo=f"{si}")

        cv2.imshow("example", im)
        key = cv2.waitKey(0)
        if key == 27:
            break


if __name__ == "__main__":
    main()
