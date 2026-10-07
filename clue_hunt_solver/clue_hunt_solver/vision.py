"""OpenCV detection helpers for clue boards, QR codes, and pillars."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from pyzbar.pyzbar import ZBarSymbol, decode as zbar_decode
except ImportError:  # Optional on development machines; required on Ubuntu VM.
    ZBarSymbol = None
    zbar_decode = None


@dataclass
class MarkerDetection:
    marker_id: int
    corners: np.ndarray
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None

    @property
    def centre(self) -> np.ndarray:
        return np.mean(self.corners, axis=0)


@dataclass
class QRDetection:
    text: str
    corners: Optional[np.ndarray]

    @property
    def centre(self) -> Optional[np.ndarray]:
        if self.corners is None:
            return None
        return np.mean(self.corners, axis=0)


@dataclass
class PillarDetection:
    colour: str
    bottom_pixel: Tuple[float, float]
    area: float


def _aruco_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    # Ubuntu 22.04 commonly ships OpenCV 4.5, whose Python API uses the
    # factory function; newer OpenCV releases expose a constructor instead.
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        parameters = cv2.aruco.DetectorParameters_create()
    else:
        parameters = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, parameters)
    return dictionary, parameters


ARUCO_DETECTOR = _aruco_detector()
QR_DETECTOR = cv2.QRCodeDetector()


def detect_markers(
    image: np.ndarray,
    marker_length: float,
    camera_matrix: Optional[np.ndarray] = None,
    distortion: Optional[np.ndarray] = None,
) -> List[MarkerDetection]:
    """Detect DICT_4X4_50 markers and optionally estimate their poses."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if hasattr(ARUCO_DETECTOR, "detectMarkers"):
        corners, ids, _ = ARUCO_DETECTOR.detectMarkers(gray)
    else:
        dictionary, parameters = ARUCO_DETECTOR
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, dictionary, parameters=parameters
        )
    if ids is None:
        return []

    half = marker_length * 0.5
    # Board convention: +X out of face, +Y reader-right, +Z up.
    object_points = np.array([
        [0.0, -half, half],
        [0.0, half, half],
        [0.0, half, -half],
        [0.0, -half, -half],
    ], dtype=np.float32)
    results: List[MarkerDetection] = []
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        pixels = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
        detection = MarkerDetection(int(marker_id), pixels)
        if camera_matrix is not None:
            dist = np.zeros(5) if distortion is None else distortion
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                pixels,
                camera_matrix,
                dist,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if success and float(np.asarray(tvec).reshape(3)[2]) > 0.0:
                detection.rvec = rvec
                detection.tvec = tvec
        results.append(detection)
    return results


def decode_qr_codes(image: np.ndarray) -> List[QRDetection]:
    """Decode all visible QR codes.

    ZBar is the primary decoder when installed.  Ubuntu 22.04's packaged
    OpenCV is often built without the QUIRC QR backend, whereas newer OpenCV
    builds include it.  Retaining the OpenCV fallback keeps the node portable.
    """
    results: List[QRDetection] = []
    if zbar_decode is not None:
        try:
            decoded_symbols = zbar_decode(image, symbols=[ZBarSymbol.QRCODE])
        except Exception:
            decoded_symbols = []
        for symbol in decoded_symbols:
            try:
                text = symbol.data.decode('utf-8').strip()
            except UnicodeDecodeError:
                continue
            polygon = getattr(symbol, 'polygon', None)
            corners = None
            if polygon:
                corners = np.asarray(
                    [[point.x, point.y] for point in polygon], dtype=np.float32
                )
            if text:
                results.append(QRDetection(text, corners))
    if results:
        return results

    try:
        ok, decoded, points, _ = QR_DETECTOR.detectAndDecodeMulti(image)
    except (cv2.error, ValueError):
        ok, decoded, points = False, (), None
    if ok and points is not None:
        for text, corners in zip(decoded, points):
            if text:
                results.append(QRDetection(text.strip(), np.asarray(corners)))
    if results:
        return results

    text, points, _ = QR_DETECTOR.detectAndDecode(image)
    if text:
        results.append(
            QRDetection(text.strip(), None if points is None else np.asarray(points))
        )
    return results


def pair_marker_with_qr(
    marker: MarkerDetection, qr_codes: Sequence[QRDetection]
) -> Optional[QRDetection]:
    """Select the QR code spatially to the reader-right of an ArUco marker."""
    if not qr_codes:
        return None
    if len(qr_codes) == 1:
        return qr_codes[0]

    right = 0.5 * (
        (marker.corners[1] - marker.corners[0])
        + (marker.corners[2] - marker.corners[3])
    )
    length = float(np.linalg.norm(right))
    if length < 1.0:
        return None
    right /= length
    best = None
    best_score = float("inf")
    for qr in qr_codes:
        centre = qr.centre
        if centre is None:
            continue
        delta = centre - marker.centre
        along = float(np.dot(delta, right))
        sideways = abs(float(np.cross(right, delta)))
        if along <= 0.2 * length:
            continue
        score = abs(along - 1.25 * length) + 2.0 * sideways
        if score < best_score:
            best, best_score = qr, score
    return best


def decode_qr_beside_marker(
    image: np.ndarray, marker: MarkerDetection
) -> Optional[QRDetection]:
    """Rectify the expected QR patch next to a marker and decode it.

    This fallback is useful when OpenCV cannot decode an oblique QR code from
    the full camera image. Board geometry fixes the QR centre 0.30 m to the
    reader's right of a 0.24 m marker.
    """
    corners = marker.corners.astype(np.float32)
    right_vec = 0.5 * ((corners[1] - corners[0]) + (corners[2] - corners[3]))
    down_vec = 0.5 * ((corners[3] - corners[0]) + (corners[2] - corners[1]))
    side = 0.5 * (np.linalg.norm(right_vec) + np.linalg.norm(down_vec))
    if side < 12.0:
        return None
    right = right_vec / max(np.linalg.norm(right_vec), 1.0)
    down = down_vec / max(np.linalg.norm(down_vec), 1.0)
    centre = marker.centre + right * (1.25 * side)
    half = 0.62 * side
    source = np.array([
        centre - right * half - down * half,
        centre + right * half - down * half,
        centre + right * half + down * half,
        centre - right * half + down * half,
    ], dtype=np.float32)
    destination = np.array(
        [[0, 0], [319, 0], [319, 319], [0, 319]], dtype=np.float32
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    patch = cv2.warpPerspective(image, transform, (320, 320))
    text, _, _ = QR_DETECTOR.detectAndDecode(patch)
    if not text:
        return None
    return QRDetection(text.strip(), source)


HSV_RANGES: Dict[str, Sequence[Tuple[np.ndarray, np.ndarray]]] = {
    "RED": (
        (np.array([0, 100, 55]), np.array([12, 255, 255])),
        (np.array([168, 100, 55]), np.array([179, 255, 255])),
    ),
    "GREEN": ((np.array([35, 70, 45]), np.array([90, 255, 255])),),
    "BLUE": ((np.array([92, 75, 45]), np.array([140, 255, 255])),),
}


def detect_pillars(image: np.ndarray) -> List[PillarDetection]:
    """Return likely coloured pillar bottom pixels for ground-plane projection."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    detections: List[PillarDetection] = []
    for colour, ranges in HSV_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, low, high))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        x, y, width, height = cv2.boundingRect(contour)
        if area < 180.0 or height < 18 or height < 1.15 * width:
            continue
        detections.append(
            PillarDetection(colour, (x + width * 0.5, y + height - 1.0), area)
        )
    return detections
