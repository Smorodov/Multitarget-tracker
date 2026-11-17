# MOT benchmark: как гонять и считать метрики

Инфраструктура для прогонов трекера на стандартных датасетах (MOT20,
VisDrone-MOT2019), экспорта результатов и расчёта метрик
(MOTA/MOTP/IDF1/IDSW). Позволяет сравнивать конфигурации детектора/трекера
и ловить регрессии.

## 1. Экспорт результатов трекера

`MultitargetTracker` пишет треки в формате MOTChallenge при задании
`--mot_res=<файл>`:

```
frame,id,x,y,w,h,conf,-1,-1,-1
```

Кадры 1-based (совпадают с нумерацией GT в MOT20/VisDrone), `x,y` — левый
верхний угол бокса, `conf` клампится в [0..1]. Флаг `--no_show=1` отключает
все окна (обязателен для батч-прогонов).

## 2. Метрики: tools/mot_eval

Отдельный инструмент (таргет `mot_eval`, опция `BUILD_MOT_EVAL`, связан с
`mtracking`):

```
mot_eval --gt <gt.txt> --res <res.txt>
         [--types <types.txt>]        # имена классов, id = номер строки + 1
         [--gt_classes ids]           # какие классы GT оценивать (через ;)
         [--gt_ignore_classes ids]    # классы-подавители FP (через ;)
         [--iou 0.5] [--width W --height H]   # клэмпинг к кадру
         [--min_area px^2]            # отсечка мелких result-боксов
         [--label имя] [--out_csv файл.csv]   # CSV с аппендом
```

Семантика повторяет py-motmetrics:

* CLEAR MOT: шаг 1 — реставрация совпадений предыдущего кадра (приоритет
  непрерывности), шаг 2 — Hungarian по оставшимся парам (IoU ≥ 0.5);
  IDSW — смена гипотезы у GT-трека.
* ID-меры (IDF1/IDTP/IDFP/IDFN): глобальное min-cost назначение
  «GT-трек × гипотеза-трек» в layout py-motmetrics (стоимость пары =
  oc + hc − 2·tps, где tps — число кадров с IoU ≥ порога у пары).
* MOTA = 1 − (FN + FP + IDSW) / numObjects (может быть отрицательной);
  MOTP — средний (1 − IoU) совпадений (конвенция py-motmetrics),
  MOTP (IoU) — средний IoU (leaderboard-конвенция).
* Классы GT, не входящие в `--gt_classes` и `--gt_ignore_classes`,
  выбрасываются целиком (не GT, не подавители). Игнор-классы никогда не
  дают FN, но давят FP.
* Подавление FP игнор-областями: IoU ≥ minIou с игнор-боксом **или центр
  result-бокса внутри игнор-бокса** (в VisDrone категория 0 — большие
  области; для маленьких боксов внутри них IoU-тест не срабатывает).

### Валидация

* `tools/mot_eval/crosscheck_motmetrics.py` — кросс-чек с py-motmetrics
  1.4.0 (`c:\python\python310\python.exe`) на MOT20-05: GT-vs-GT и
  perturbed GT с инъекцией FN/FP/IDSW. Все метрики совпадают в пределах
  округления вывода (4 знака).
* `tools/mot_eval/tests_mot_eval.py` — 9 синтетических кейсов с ручным
  расчётом (perfect, FN, FP, ID switch, подавление игнор-областью по IoU и
  по центру, выброшенный класс, клэмпинг, min_area).

## 3. Прогон бенчмарка: data/mot_bench.ps1

```
powershell -ExecutionPolicy Bypass -File data\mot_bench.ps1 -Dataset mot20 -OutDir <dir>
powershell -ExecutionPolicy Bypass -File data\mot_bench.ps1 -Dataset visdrone -OutDir <dir>
powershell -ExecutionPolicy Bypass -File data\mot_bench.ps1 -Dataset all -OutDir <dir> [-EvalOnly]
```

* `-EvalOnly` — пересчитать метрики по уже сохранённым `<seq>.txt`
  (быстрая итерация по настройкам оценки без перепрогона трекера).
* Результаты: `<dir>\<seq>.txt` (MOT-формат), `<dir>\<dataset>.csv`
  (по строке на последовательность), логи запусков `<seq>.txt.run.log/err`.
* Датасеты: MOT20 в `C:\work\dataset\motchallenge` (img1 + seqinfo.ini +
  MOT20Labels), VisDrone2019-MOT-val в `C:\work\dataset\visdrone\mot_2019`.

## 4. Конфигурации детекции

| | settings_mot20.ini | settings_visdrone.ini |
|---|---|---|
| Модель | YOLO11s (COCO), 736×1280 | YOLOv8l-visdrone, 960×1280 |
| Бэкенд | TensorRT (`detector_backend=5`) | OpenCV DNN CUDA (`=6`) |
| Классы | white_list=person | все 10 классов VisDrone |
| conf | 0.3 | 0.25 |

Обе модели экспортированы из PT (ultralytics 8.4.90, opset 13) — исходники:
`C:\work\yolo\ultralytics\weights` (mshamrai/yolov8l-visdrone с HuggingFace,
`yolo11s.pt`). Замечание: прямоугольный вход YOLO11 не работает в OpenCV
DNN (NaryEltwise с транспонированными шейпами), поэтому MOT20 гоняется на
TensorRT; VisDrone-модель (YOLOv8) работает в OCV DNN CUDA после фикса H23
(W/H автодетекта входа).

## 5. Конфигурация оценки

| | MOT20 | VisDrone val |
|---|---|---|
| eval | класс 1 (Pedestrian) | классы 1–10 |
| ignore (давят FP) | 2,7,8 + 9,10,11,12 | 0, 11 |
| выброшены | 3,4,5,6 | — |
| min_area | 0 | 800 |
| кадр | из seqinfo.ini | из первого кадра |

Игнор-набор MOT20 следует конвенции TrackEval (дистракторы/окклюдеры
подавляют FP). `--min_area=800` для VisDrone — практика против потока
детекций объектов, отсутствующих в MOT-разметке (модель уверенно видит
далёкие мелкие объекты: медиана conf у таких FP ≈ 0.41 — порогом conf это
не лечится).

## 6. Базовые результаты (baseline1, 2026-11-16/17)

MOT20 train (TensorRT, yolo11s/person):

| Seq | GT | MOTA | IDF1 | IDSW | MOTP (IoU) |
|---|---|---|---|---|---|
| MOT20-01 | 19870 | 0.255 | 0.290 | 509 | 0.714 |
| MOT20-02 | 154742 | 0.292 | 0.248 | 2556 | 0.721 |
| MOT20-03 | 313658 | 0.150 | 0.239 | 2595 | 0.696 |
| MOT20-05 | 646344 | 0.094 | 0.124 | 4252 | 0.722 |

VisDrone2019-MOT-val (OCV DNN CUDA, yolov8l-visdrone, min_area=800):

| Seq | GT | MOTA | IDF1 | IDSW | MOTP (IoU) |
|---|---|---|---|---|---|
| uav0000086_00000_v | 23097 | -0.312 | 0.301 | 331 | 0.699 |
| uav0000117_02622_v | 17279 | -4.128 | 0.091 | 837 | 0.706 |
| uav0000137_00458_v | 24088 | -2.317 | 0.250 | 1460 | 0.727 |
| uav0000182_00000_v | 17671 | -0.963 | 0.298 | 239 | 0.718 |
| uav0000268_05773_v | 13578 | -0.500 | 0.302 | 95 | 0.718 |
| uav0000305_00000_v | 6391 | -2.748 | 0.272 | 355 | 0.785 |
| uav0000339_00001_v | 12028 | -1.727 | 0.208 | 334 | 0.719 |

Чтение: MOTP 0.70–0.78 на обеих базах — локализация боксов хорошая.
Узкое место — recall/FP-дисбаланс детекции:

* MOT20: COCO-модель на плотных толпах находит 30–55% GT-боксов
  (точки роста: crowd-обученная модель, yolo11m/x, ниже conf).
* VisDrone: детекции объектов вне разметки GT (минус в MOTA), усиление —
  min_area/порог conf/фильтрация статики; для осмысленных сравнений
  конфигураций между собой метрики при фиксации одного и того же набора
  параметров оценки полностью пригодны.

## 7. Повторение/итерация

1. Поменять `data/settings_*.ini` (модель/бэкенд/пороги) или настройки
   трекера (раздел `[tracking]`).
2. Полный прогон: `-Dataset all` (MOT20 ~6 мин + VisDrone ~5.5 мин на RTX
   4070); только метрики: `-EvalOnly`.
3. Сравнить `<OutDir>\<dataset>.csv` со строками из этого документа
   (baseline1) — регрессии видны по MOTA/IDF1/IDSW.
