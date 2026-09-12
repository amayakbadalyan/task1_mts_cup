#!/usr/bin/env python3

"""СТАРТЕР — Уровень 1 «Квалификация».

Это ЕДИНСТВЕННЫЙ файл, который ты редактируешь и сдаёшь. Всё остальное (мир, робот,
локомоция go2_api, судья) фиксировано и одинаково для всех.

Сейчас робот просто медленно едет вперёд и на первой же стене упрётся — это НЕ решение,
а каркас. Твоя задача: довести робота по чекпоинтам до финиша, используя сенсоры
(лидар/одометрию), а не жёстко прописанные координаты. Описание — level1/TASK.md, API — API.md.
"""

import math
import os
import sys
import numpy as np
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "common", "locomotion"))
from go2_api import Go2

def analyze_lidar(lidar, k, r=0.6, n=6):
    """
    lidar: list из 24 слоёв, каждый слой — list из 360 float-расстояний.
    k: порог количества точек.
    r: радиус в метрах.
    n: количество центральных слоёв.
    """
    num_layers = len(lidar)
    if num_layers != 24:
        raise ValueError("Ожидается 24 слоя лидара")

    # Центральные n слоёв. Для 24 и n=6: индексы 9..14.
    start = (num_layers - n) // 2
    end = start + n
    central_layers = lidar[start:end]

    total = 0
    sector_counts = [0] * 8  # 8 секторов по 45 градусов

    for layer in central_layers:
        for degree, dist in enumerate(layer):
            # degree направление в градусах: 0..359
            if dist <= r:
                total += 1
                sector = degree // 45  # 0..7
                sector_counts[sector] += 1

    if total <= k:
        return {
            "triggered": False,
            "total": total,
            "best_sector": None,
            "best_direction": None,
            "best_count": None,
            "sector_counts": None,
        }

    best_sector = max(range(8), key=lambda s: sector_counts[s])
    best_count = sector_counts[best_sector]

    labels = [
        "0-45°",
        "45-90°",
        "90-135°",
        "135-180°",
        "180-225°",
        "225-270°",
        "270-315°",
        "315-360°",
    ]

    return {
        "triggered": True,
        "total": total,
        "best_sector": best_sector,
        "best_direction": labels[best_sector],
        "best_count": best_count,
        "sector_counts": sector_counts,
    }


TARGET = np.array([147, 243, 255])   # R=147, G=243, B=255
TOL = 30                              # допуск на канал
MIN_AREA = 50                         # минимальная площадь группы


def get_x_offset(rgb, target_rgb, tol=30, min_area=50):
    '''
    Находит смещение по оси X на изображении от центра экрана до bbox таргета в пикселях.
    Работает на изображении, уменьшенном в 2 раза, и умножает результат обратно на 2.
    '''
    rgb = rgb[::2, ::2]
    min_area = max(1, min_area // 4)

    H, W = rgb.shape[:2]
    img_cx = W // 2

    diff = np.abs(rgb.astype(np.int16) - target_rgb)
    mask = np.all(diff <= tol, axis=2)


    def erode(m):
        p = np.pad(m, 1, constant_values=False)
        return (p[:-2, 1:-1] & p[2:, 1:-1] &
                p[1:-1, :-2] & p[1:-1, 2:] & p[1:-1, 1:-1])


    def dilate(m):
        p = np.pad(m, 1, constant_values=False)
        return (p[:-2, 1:-1] | p[2:, 1:-1] |
                p[1:-1, :-2] | p[1:-1, 2:] | p[1:-1, 1:-1])

    mask = dilate(erode(mask))

    visited = np.zeros_like(mask, dtype=bool)
    best = None

    for y0 in range(H):
        for x0 in range(W):
            if not mask[y0, x0] or visited[y0, x0]:
                continue
            q = deque([(y0, x0)])
            visited[y0, x0] = True
            area = 0
            minx = maxx = x0
            miny = maxy = y0
            while q:
                y, x = q.popleft()
                area += 1
                if x < minx: minx = x
                if x > maxx: maxx = x
                if y < miny: miny = y
                if y > maxy: maxy = y
                for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W \
                            and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
            if area >= min_area:
                if best is None or area > best[0]:
                    best = (area, minx, miny, maxx, maxy)

    if best is None:
        return None

    _, minx, miny, maxx, maxy = best
    bx = (minx + maxx) // 2
    return (bx - img_cx) * 2    


BASE_SPEED = 0.35 # Скорость чтобы не упасть
PI = math.pi
TIME_SEARCH = 3 # Время поиска
got_target = 0 #Найденные цели

def main():
    robot = Go2()
    i = 0
    time = -1
    while robot.step():
        scan = robot.lidar()
        lidar_frame = [scan[i:i+360] for i in range(0, len(scan), 360)]
        yaw, pitch, roll = robot.imu()
        x, y, z = robot.position()
        timestamp = robot.time()

        dspeed_x = 0
        dspeed_y = 0
        dangle = 0

        i += 1
        if i % 10 == 0:
            founded_obst = analyze_lidar(lidar_frame, 0)
            print(founded_obst)
            if founded_obst['triggered']:
                dangle = founded_obst['best_sector'] * PI / 4 + PI / 8 + PI
            else:
                dangle = 0
        dspeed_x = BASE_SPEED * math.cos(dangle)
        dspeed_y = BASE_SPEED * math.sin(dangle)

        if i % 1000 == 0:
            #print(scan)
            pass
        
        arr = np.frombuffer(robot.image(), np.uint8).reshape(240, 320, 4)
        rgb = arr[..., :3].copy()

        #with open(r"C:\Users\amaya\OneDrive\Desktop\frame.bin", "wb") as f:
        #    f.write(robot.image())

        dx = get_x_offset(rgb, TARGET)

        #print(dangle, dspeed_y, dspeed_x)
        if dx is not None:
            robot.drive(vx=dspeed_x, vy=dspeed_y, vyaw=0.0 - 1.2 * dx / 320)
            time = -1
        else:
            if time < 0:
                time = timestamp
            if timestamp < time + TIME_SEARCH:
                robot.drive(vx=dspeed_x, vy=dspeed_y, vyaw=0.0)
            else:
                robot.drive(vx=0, vyaw=0.5)


if __name__ == "__main__":
    main()