import copy
import os
from typing import List, Optional, Tuple

import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt

from ..bbox import LiDARInstance3DBoxes
import gc

__all__ = ["visualize_camera", "visualize_lidar", "visualize_map", "visualize_map_carla"]


OBJECT_PALETTE = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 69, 0),
    "trailer": (255, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
}

MAP_PALETTE = {
    "drivable_area": (166, 206, 227),
    "road_segment": (31, 120, 180),
    "road_block": (178, 223, 138),
    "lane": (51, 160, 44),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "road_divider": (202, 178, 214),
    "lane_divider": (106, 61, 154),
    "divider": (106, 61, 154),
}


def visualize_camera(
    fpath: str,
    image: np.ndarray,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    scores: Optional[np.ndarray] = None,  # Añadido para recibir las puntuaciones
    transform: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: float = 4,
) -> None:
    canvas = image.copy()
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    if bboxes is not None and len(bboxes) > 0:
        corners = bboxes.corners
        num_bboxes = corners.shape[0]

        coords = np.concatenate(
            [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
        )
        transform = copy.deepcopy(transform).reshape(4, 4)
        coords = coords @ transform.T
        coords = coords.reshape(-1, 8, 4)

        indices = np.all(coords[..., 2] > 0, axis=1)
        coords = coords[indices]
        labels = labels[indices]

        if scores is not None:
            scores = scores[indices]  # Filtrar las puntuaciones en el mismo orden
            #print(scores)
        
        indices = np.argsort(-np.min(coords[..., 2], axis=1))
        coords = coords[indices]
        labels = labels[indices]
        #print(scores)
        if scores is not None:
            scores = scores[indices]  # Ordenar las puntuaciones en el mismo orden
        #print(scores)

        coords = coords.reshape(-1, 4)
        coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= coords[:, 2]
        coords[:, 1] /= coords[:, 2]

        coords = coords[..., :2].reshape(-1, 8, 2)
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            score = scores[index] if scores is not None else None
            label_text = f"{name} {score:.2f}" if score is not None else name

            # Dibujar las líneas de las bboxes
            for start, end in [
                (0, 1),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 5),
                (3, 2),
                (3, 7),
                (4, 5),
                (4, 7),
                (2, 6),
                (5, 6),
                (6, 7),
            ]:
                cv2.line(
                    canvas,
                    coords[index, start].astype(int),
                    coords[index, end].astype(int),
                    color or OBJECT_PALETTE[name],
                    thickness,
                    cv2.LINE_AA,
                )

            # Dibujar el texto con la clase y la puntuación, con un fondo transparente al estilo BEVFusion
            text_position = coords[index, 0].astype(int)  # Usar la primera esquina como posición del texto
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,  # Tamaño de letra ajustado para estilo BEVFusion
                2,    # Grosor de la fuente
            )

            # # Crear un rectángulo transparente para el fondo del texto
            # overlay = canvas.copy()
            # cv2.rectangle(
            #     overlay,
            #     (text_position[0], text_position[1] - text_height - baseline),
            #     (text_position[0] + text_width - 80, text_position[1]),
            #     (200, 200, 200),
            #     -1,  # Relleno del rectángulo
            # )
            # alpha = 0.6  # Nivel de transparencia
            # cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

            canvas = draw_transparent_rect(
                canvas,
                (text_position[0] - 2, text_position[1] - text_height - 8),  # Posición ajustada para encajar el texto
                int(text_width *1.05),  # Ancho exacto del texto con un pequeño margen
                text_height + 12,  # Alto del texto con margen y baseline
                color=(200, 200, 200),  # Color RGB (200, 200, 200)
                alpha=0.6  # Transparencia del 60%
            )

            # Añadir el texto encima del rectángulo
            cv2.putText(
                canvas,
                label_text,
                (text_position[0] + 5, text_position[1] - 5),  # Ajustar el texto dentro del rectángulo
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,  # Tamaño de letra ajustado
                OBJECT_PALETTE[name],  # Texto en color negro para mejor visibilidad sobre el fondo naranja
                2,  # Grosor del texto
                cv2.LINE_AA,
            )

        canvas = canvas.astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)


def draw_transparent_rect(image, position, width, height, color, alpha=0.6):
    overlay = image.copy()
    output = image.copy()
    cv2.rectangle(overlay, position, (position[0] + width, position[1] + height), color, -1)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output



def visualize_lidar(
    fpath: str,
    lidar: Optional[np.ndarray] = None,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    color: Optional[Tuple[int, int, int]] = None,
    radius: float = 8, #15
    thickness: float = 14, #25
) -> None:
    #fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))
    fig = plt.figure(figsize=(50, 50))

    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if lidar is not None:
        if lidar.shape[0] > 1000:  # Limitar a un máximo de 10,000 puntos
            lidar = lidar[np.random.choice(lidar.shape[0], 1000, replace=False)]
        plt.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=radius,
            c="white",
        )

    if bboxes is not None and len(bboxes) > 0:
        coords = bboxes.corners[:, [0, 3, 7, 4, 0], :2]
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            plt.plot(
                coords[index, :, 0],
                coords[index, :, 1],
                linewidth=thickness,
                color=np.array(color or OBJECT_PALETTE[name]) / 255,
            )

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    fig.savefig(
        fpath,
        dpi=10,
        facecolor="black",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()
    gc.collect()


def visualize_map(
    fpath: str,
    masks: np.ndarray,
    *,
    classes: List[str],
    background: Tuple[int, int, int] = (240, 240, 240),
) -> None:
    assert masks.dtype == np.bool, masks.dtype

    canvas = np.zeros((*masks.shape[-2:], 3), dtype=np.uint8)
    canvas[:] = background

    for k, name in enumerate(classes):
        if name in MAP_PALETTE:
            canvas[masks[k], :] = MAP_PALETTE[name]
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)

def visualize_map_carla(
    fpath: str,
    masks: np.ndarray,
    *,
    classes: List[str],
    background: Tuple[int, int, int] = (240, 240, 240),
) -> None:
    assert masks.dtype == np.bool, masks.dtype

    canvas = np.zeros((*masks.shape[-2:], 3), dtype=np.uint8)
    canvas[:] = background

    for k, name in enumerate(classes):
        if name in CARLA_MAP_PALETTE:
            canvas[masks[k], :] = CARLA_MAP_PALETTE[name]
    
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)
