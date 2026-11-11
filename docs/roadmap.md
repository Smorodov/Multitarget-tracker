# Roadmap: найденные проблемы и план работ

> **Дата анализа:** 2026-11-10.
> **Метод:** статический разбор исходного кода (без сборки и запуска).
> **Область:** весь репозиторий `Multitarget-tracker` (`src/`, `example/`,
> `async_detector/`, `setup.py`, CMake, CI).
>
> Нумерация `файл:строка` соответствует состоянию на дату анализа.
> Часть дефектов проявляется только при определённых опциях сборки/конфигурации;
> это указано в карточках. Оценки «сработает/не сработает» — результат
> статического анализа, а не воспроизведения.

> **Статус на 2026-11-11:** C1 (фабрика детекторов), C2 (LAPJV), C7
> (инициализация TensorRT), C8 (`async_detector`), C11 (ROI гистограмм),
> C12 (обрезка траектории) и C13 (VIBE RNG/`% 0`) исправлены в рабочем
> дереве; C2, C7, C8, C11, C12 и C13 — в этих правках, см. карточки.
> Остальные пункты актуальны.

## 1. Легенда приоритетов

| Приоритет | Значение | Что делать |
|---|---|---|
| **P0** | Падение процесса, неопределённое поведение (UB), порча памяти, нерабочая заявленная функция, неразыменовываемый `nullptr` | Исправить в первую очередь, до любых релизов |
| **P1** | Неверный результат трекинга/детекции, зависание, исключение при валидной конфигурации, утечка/рост памяти | Исправить в ближайшем релизе |
| **P2** | Надёжность, производительность, переносимость сборки, неудобный/небезопасный API | Плановые работы |
| **P3** | Документация, мёртвый код, косметика, стиль | По возможности |

## 2. Сводка

### 2.1 P0 — критично

| ID | Кратко | Где |
|---|---|---|
| **C1** | Фабрика детекторов разыменовывает `nullptr` при неподдерживаемом детекторе | `src/Detector/BaseDetector.cpp:61` |
| **C2** | LAPJV: неверная граница сканирования и выход за границу `collist` на прямоугольной матрице | `src/Tracker/LAPJV_algorithm/lap.cpp:182-215` |
| **C3** | `exit(-1)`/`assert(0)` в библиотеке при «плохом» значении cost | `src/Tracker/BaseTracker.cpp:604-608` |
| **C4** | Выход за границы heat map при отрисовке (запись в память) | `example/CarsCounting.cpp:442-456` |
| **C5** | Некорректный ROI при вставке карты в кадр → исключение/крэш | `example/CarsCounting.cpp:340-343` |
| **C6** | `ParseDFINE` читает `int64_t` из blob без проверки типа и порядка выходов; `DFINE_IS` — `assert(0)` | `src/Detector/OCVDNNDetector.cpp:1055-1093, 1104` |
| **C7** | TensorRT: неудачная десериализация engine считается успехом; нет веток YOLOV3/V4 в `YoloONNXImpl::Init`; `assert` при ошибке | `src/Detector/tensorrt_onnx/YoloONNX.cpp:87-149`, `class_detector.cpp:55-203` |
| **C8** | `async_detector`: гонка на `bool stopFlag`, разыменование `nullptr` детектора, падение при отсутствии `logs/` | `async_detector/AsyncDetector.cpp:46,62-64,387-394` |
| **C9** | Парсеры YOLO читают `classNames.size()` float-значений без проверки размеров выхода → чтение за границей blob | `src/Detector/OCVDNNDetector.cpp:586-589` и аналоги |
| **C10** | `ProcessInputAspectRatio` обходит `inputB` (из engine), а не фактический размер батча → OOB чтение `sampleImages[b]` | `src/Detector/tensorrt_onnx/YoloONNX.cpp:382-431` |
| **C11** | Гистограммы по ROI без проверки границ кадра (дефолтная конфигурация трекера) → `cv::Exception` | `src/Tracker/BaseTracker.cpp:645-646` |
| **C12** | `maxTraceLength <= 0` → `Trace::at(0)` на пустом векторе → `std::out_of_range` | `src/Tracker/track.cpp:333-340` |
| **C13** | VIBE: гонка на общем RNG в OpenMP-цикле; `updateFactor == 0` → деление по модулю на 0 | `src/Detector/vibe_src/vibe.cpp:27,84,117` |
| **C14** | Python-обёртка не компилируется и возвращает numpy-массивы на освобождённой памяти | `src/python_bind/mtracker.cpp:11,112,256-259,393` |

### 2.2 P1 — высокая важность

| ID | Кратко | Где |
|---|---|---|
| **H1** | Неверная нормализация при отсутствии ReID-эмбеддинга (`dist /= weight`) | `src/Tracker/BaseTracker.cpp:556-566` |
| **H2** | `GetVelocity` возвращает не те индексы состояния для 6- и 10-мерных моделей | `src/Tracker/Kalman.cpp:1206-1244` |
| **H3** | Mahalanobis: не та ковариация, несовпадение размерностей, отрицательные cost | `src/Tracker/track.cpp:184-205`, `Kalman.cpp:1247-1270` |
| **H4** | Угол эллипса в радианах записывается в `RotatedRect` (градусы), затем смешивается в `IsInsideArea` | `src/Tracker/track.cpp:439-483` |
| **H5** | «Inertia correction» учитывает пустую детекцию (0,0) на потерянных кадрах | `src/Tracker/Kalman.cpp:769-782,948-963` |
| **H6** | Матрица перехода `CreateLinearAcceleration(Rect)` смешивает `aw`/`ah` в координаты `x`/`y` | `src/Tracker/Kalman.cpp:307-319` |
| **H7** | Не реализованы/ассертятся комбинации: rrect + acceleration, UKF + FilterRRect, UKF + FilterRect (`GetRectPrediction`) | `Kalman.cpp:383-387,812-838,1090-1105` |
| **H8** | ByteTrack: `staticTime` всегда 0; `removed_stracks_` растёт неограниченно; `GetRemovedTracks` не очищает вектор | `src/Tracker/byte_track/BYTETracker.cpp:25-51,214-240` |
| **H9** | «Статика»: `min_static_time` (секунды) используется как число кадров; `max_speed_for_static` игнорируется | `src/Tracker/track.cpp:516-580`, `BaseTracker.cpp:415` |
| **H10** | `CarsCounting` игнорирует секцию `[tracking]` из `--settings` | `example/CarsCounting.cpp:249-284` |
| **H11** | Геопривязка: гонка `m_geoBindingsRead`, деление на `w = 0`, отсутствие проверки гомографии | `CarsCounting.cpp:68-72,241-245`, `CarsCounting.h:396-410,507-521` |
| **H12** | `RoadLine`: `(m_pt1.y + m_pt1.y)/2` вместо `m_pt2.y`; чтение неинициализированного `intersectPt` | `example/CarsCounting.h:590-735` |
| **H13** | `Esc` в `SyncProcess` не завершает воспроизведение (break только внутреннего цикла батча) | `example/VideoExample.cpp:220-241` |
| **H14** | `Frame::GetUMatGray` сравнивает thread id с полем серого канала вместо BGR | `example/VideoExample.h:119-135` |
| **H15** | `m_staticStartTime` не инициализируется → `staticTime` ≈ 1.7e9 секунд у нового трека | `src/Tracker/track.cpp:616-623`, `track.h:196` |
| **H16** | `setup.py`/`demo.py` несовместимы с актуальным API и Python 3.12+ (`distutils`) | `setup.py:8,138,176`, `demo.py:73,86` |
| **H17** | `ParseTrackerSettings`: INI с синтаксической ошибкой считается успешно прочитанным | `src/Tracker/TrackerSettings.cpp:16` |
| **H18** | `GetCrops` может зависнуть при `maxCropRatio ≈ 0` (`stepX/stepY == 0`) | `src/Detector/BaseDetector.h:229-277` |
| **H19** | `detections[0]/[1]/[2]` в парсерах без проверки `detections.size()` → OOB `std::vector` | `src/Detector/OCVDNNDetector.cpp:560,620,929,1057` |
| **H20** | `CTracker::Update(..., DistType)` фактически влияет только на выбор ENU-порога | `src/Tracker/BaseTracker.cpp:119-137` |
| **H21** | `TrackerSettings` по умолчанию использует `DistHist` и backend TensorRT — небезопасные/неожиданные дефолты | `TrackerSettings.h:23,173,232-240` |
| **H22** | `TypeConverter::Type2Str` без проверки границ; `Str2Type` неограниченно расширяет глобальный список | `src/mtracking/object_types.h:14-33` |

### 2.3 P2/P3 — плановые

Ниже — сводные таблицы; подробные карточки для них не расписывались, но
направление исправления указано.

| ID | Приоритет | Кратко | Где |
|---|---|---|---|
| **M1** | P2 | Каждый кадр копируется в `m_prevFrame`, но параметр `prevFrame` нигде не используется (лишняя копия полного кадра) | `src/Tracker/BaseTracker.cpp:139`, `track.cpp:269-400` |
| **M2** | P2 | `CheckStatic` клонирует полный кадр на каждый новый статический объект (`m_staticFrame` дальше не читается) | `src/Tracker/track.cpp:545` |
| **M3** | P2 | `CalcEmbeddins` каждый кадр выделяет `regions.size()` объектов; `GetTracks` копирует все траектории | `BaseTracker.cpp:161-162,622-674` |
| **M4** | P2 | Поиск непривязанных регионов через `std::find` → O(T·R) на кадр | `BaseTracker.cpp:363-393` |
| **M5** | P2 | Обновление треков в `#pragma omp parallel for` с общими `cv::UMat` и визуальными трекерами OpenCV — потенциальные гонки/нестабильность | `BaseTracker.cpp:400-428` |
| **M6** | P2 | `DrawFilledRect`: `color[3]` при 4-канальном кадре и арифметика указателя при `rect.x >= cols` | `src/mtracking/defines.h:361-385` |
| **M7** | P2 | `nms/nms2/nms3`: деление на нулевую площадь объединения → NaN | `src/mtracking/nms.h:48-49,116-117,191-192` |
| **M8** | P2 | `WidthDist`/`HeightDist`: деление на 0 при нулевом размере | `src/Tracker/track.cpp:490-509` |
| **M9** | P2 | `TrackingObject::IsRobust` при пустой траектории: `size()-1` (underflow) и `0/0` → NaN | `src/Tracker/trajectory.h:260-284` |
| **M10** | P2 | `LeastSquarespoly2/2`: параметр `count` используется как конечный индекс; `sqrt` от отрицательного → NaN | `src/Tracker/trajectory.h:304-396` |
| **M11** | P2 | `BaseDetector::UpdateIgnoreMask`: отрицательный размер ROI при выходе за кадр | `src/Detector/BaseDetector.h:118-132` |
| **M12** | P2 | `BackgroundSubtract::Init` «успешно» возвращает `true` после fallback на VIBE; ошибки не видны вызывающему | `src/Detector/BackgroundSubtract.cpp:19-197` |
| **M13** | P2 | `std::stoi/stof` по значениям INI/KeyVal без `try/catch` → необработанное исключение | `MotionDetector.cpp:39`, `TrackerSettings.cpp`, `ONNXTensorRTDetector.cpp:50-63` |
| **M14** | P2 | `m_writeEachNFrame == 0` → деление на 0 при логировании | `example/FileLogger.h:113,149` |
| **M15** | P2 | `logs/` не создаётся → `rotating_file_sink` бросает исключение при старте | `example/VideoExample.cpp:55`, `AsyncDetector.cpp:46` |
| **M16** | P2 | `CarsCounting::RemoveLine` всегда возвращает `false`, даже удалив линию | `example/CarsCounting.cpp:391-401` |
| **M17** | P2 | `GeoParams::Pix2MapParams::Init`: при неудаче `m_toPix` остаётся неинициализированной | `example/CarsCounting.h:507-521` |
| **M18** | P2 | `VideoExample::AsyncProcess`: при `batch_size > 1` отображается/обрабатывается только первый кадр батча | `example/VideoExample.cpp:309-328` |
| **M19** | P2 | `AsyncDetector` измеряет не время обработки, а ожидание в очереди | `async_detector/AsyncDetector.cpp:77-90` |
| **M20** | P2 | Корневой CMake: переопределение `CMAKE_BINARY_DIR`, сброс пользовательских флагов, `-march=native` в Release | `CMakeLists.txt:7-10,23,33` |
| **M21** | P2 | MSVC-ветка собирает статические библиотеки, `mtracking` не линкует OpenCV; поведение отличается от GCC | `src/Tracker/CMakeLists.txt:82-99` |
| **M22** | P2 | Нет `Threads::Threads` для `std::thread`/spdlog; OpenMP подключён через флаги, а не target | `CMakeLists.txt:12-16` |
| **M23** | P2 | Мёртвые include-пути (`Detector/Subsense`, `Detector/darknet/...`), несуществующие CMake-опции в `setup.py` | `example/CMakeLists.txt:26`, `src/CMakeLists.txt:19-20`, `setup.py:151-152` |
| **M24** | P2 | `GetTracks`/`GetRemovedTracks` — неконстантный/несогласованный контракт (BYTE не очищает, CTracker очищает) | `BaseTracker.h:36-43`, `BYTETracker.cpp:43-51` |
| **M25** | P2 | `m_velocity` документирован как «pixels/sec», но это приращение за шаг фильтра (dt не равен реальному интервалу) | `trajectory.h:207`, `Kalman.cpp:1206` |
| **M26** | P2 | `m_staticTime` в `TrackingObject` — «время с последнего сброса», а не «время статичности» | `track.cpp:616-623` |
| **M27** | P3 | `CTrack` ctor №1 смещает первую точку траектории на `height/2`, ctor №2 — нет | `track.cpp:55` vs `110` |
| **M28** | P3 | `CreateLinear(RRect)` резервирует `m_initialRects` вместо `m_initialRRects` | `Kalman.cpp:230` |
| **M29** | P3 | `CreateLinear(Rect)`: `transitionMatrix(3,7) = dt/10` — несогласованно с остальными состояниями | `Kalman.cpp:107` |
| **M30** | P3 | Средний угол при инициализации rrect-модели — арифметическое среднее (неверно у ±90°) | `Kalman.cpp:1069-1076` |
| **M31** | P3 | `TrackID::ID2Module(0)` → деление на 0; `Str2ID` бросает `std::invalid_argument` | `defines.h:53-64` |
| **M32** | P3 | `TrackerSettings::EmbeddingParams` использует `assert` в конструкторе (в Release не проверяет) | `TrackerSettings.h:216-224` |
| **M33** | P3 | `type_caster<cv::Mat>::load` игнорирует strides/continuity numpy-массива | `src/python_bind/mtracker.cpp:38-75` |
| **M34** | P3 | CVAT: жёстко `+00:00` при `localtime`; нет экранирования XML; `stop_frame = -1` при `framesCount = 0` | `example/cvatAnnotationsGenerator.h:183,199-208` |
| **M35** | P3 | `ResultsLog`: файл не открывается при имени ≤ 5 символов; CSV без ID трека | `example/FileLogger.h:29-38,109-125` |
| **M36** | P3 | Расхождение версий: CMake `1.1.0` vs `setup.py`/`__version__` `1.0.1` | `CMakeLists.txt:3`, `setup.py:194`, `mtracker.cpp:427` |
| **M37** | P3 | README: `#include <mtracking/Ctracker.h>` (файла нет), устаревшие разделы/опции | `README.md:112` |
| **M38** | P3 | `data/settings.ini`: опечатка `inference_precison`; устаревшие `TrackMIL...`, `MatchBipart`, «speed in pixels» | `data/settings.ini:50,76-95,138` |
| **M39** | P3 | `data/settings_yolov8.ini` содержит абсолютные пути разработчика (`C:/work/...`) | `data/settings_yolov8.ini:29-31` |
| **M40** | P3 | CI: только дефолтная сборка; нет тестов, Python, TensorRT, UKF, CarsCounting | `.github/workflows/*.yml` |
| **M41** | P3 | `defines.h` тянет `opencv2/opencv.hpp` целиком, использует макросы `El_t`/`Mat_t`, не включает `<chrono>` напрямую | `src/mtracking/defines.h:1-29` |
| **M42** | P3 | `TrackingObject::IsRobust` меняет `mutable m_lastRobust` в константном методе — не потокобезопасно | `trajectory.h:214,260` |
| **M43** | P3 | `GetTrajectory` возвращает `vector<cv::Point>` (округление), теряя субпиксельную точность | `trajectory.h:290-298` |
| **M44** | P3 | `GetTracks` в `BYTETracker` строит `RotatedRect` из 3 точек — NaN при вырожденном прямоугольнике | `BYTETracker.cpp:34`, `defines.h:185-193` |
| **M45** | P3 | `AsyncDetector`/`VideoExample` дублируют код логгеров, отрисовки, CLI | `async_detector/*`, `example/*` |
| **M46** | P2 | Авто-определение входа по 4D-выходу (`DetectionOutput`) даёт `N×7` вместо размера сети; из-за дефолта `608` ветка обычно мертва | `OCVDNNDetector.cpp:282-327`, `OCVDNNDetector.h:68-69` |
| **M47** | P2 | `VerifyOutputAspectRatio` удаляет имена выходов из `m_params` во время обхода, а `m_outpuDims` не привязаны к именам → после удаления данные читаются как чужие тензоры | `YoloONNX.cpp:29-66,479-505` |
| **M48** | P2 | `assert(0)` как обработка ошибок конфигурации (файл классов, неизвестный `net_type`) в детекторах | `OCVDNNDetector.cpp:210`, `ONNXTensorRTDetector.cpp:138,144` |

---

## 3. P0 — подробно

### C1. Фабрика детекторов разыменовывает `nullptr`

**Где:** `src/Detector/BaseDetector.cpp:15-64` (ключевая строка — 61).

**Симптом:** segfault/`Access violation` при попытке создать детектор, который
не скомпилирован или не существует:

* `tracking::ONNX_TensorRT` при `BUILD_ONNX_TENSORRT=OFF` (строки 49-55:
  печатается сообщение, но `detector` остаётся пустым);
* `tracking::DNN_OCV` при `USE_OCV_DNN=OFF` (строки 43-47 — `case` вообще
  отсутствует, попадаем в `default`);
* любое неизвестное/новое значение `Detectors` (например, из INI
  `detector_backend`, которое никто не валидирует: `CarsCounting.cpp:217`).

**Причина:** после `switch` безусловно вызывается `detector->Init(config)`.

**Исправление:**

```cpp
if (!detector)
{
    std::cerr << "Detector type " << static_cast<int>(detectorType)
              << " is not available in this build" << std::endl;
    return nullptr;
}
if (!detector->Init(config))
    detector.reset();
return detector;
```

**Проверка:** unit-тест «создание ONNX_TensorRT в сборке без TensorRT
возвращает `nullptr`, не падает»; то же для `DNN_OCV` без DNN.

---

### C2. LAPJV: прямоугольная задача и выход за границу `collist`

**Где:** `src/Tracker/LAPJV_algorithm/lap.cpp` (`lap()` и `lapSquare()`).

**Статус:** ✅ исправлено (2026-11-11).

**Симптом:** при `числе треков != числе детекций` (типичный случай) алгоритм
просматривал не все столбцы (`dimRows` вместо `dimCols`) и мог прочитать
`collist[dimCols]` за границей `std::vector`. Однако одной замены границ
недостаточно: LAPJV — квадратный алгоритм, и на прямоугольной матрице он
выдаёт субоптимальные назначения. Брутфорс это подтвердил: 7146 расхождений
из 34154 на матрицах 2×3 и подобных.

**Причина:** код адаптирован из квадратной версии, а прямоугольная задача
решалась «как есть», без фиктивных строк/столбцов.

**Исправление:**

* `lapSquare()` — прежний алгоритм с корректными границами `dimCols` для
  квадратной матрицы;
* `lap()` — обёртка: при `rows != cols` дополняет матрицу нулевыми
  фиктивными строками/столбцами до `dim = max(rows, cols)`, решает
  квадратную задачу и отбрасывает фиктивные назначения (`-1`);
* добавлены проверки размеров входных/выходных векторов, защита
  `up >= dimCols` (на случай нефинитных cost) и согласованный возврат
  «нет назначения» (`-1`) вместо чтения за границей.

**Проверка:** брутфорс-тест (перебор всех инъективных назначений) на
34 154 матрицах размеров 1×1…7×8, включая нулевые и одинаковые стоимости:
0 расхождений. `mtracking` собирается в конфигурации проекта (MSVC Release).
Тест: `build/lapjv_check/test.cpp` (каталог `build/` в `.gitignore`).

**Важно:** `TrackerSettings::m_matchType` по умолчанию — `MatchLAPJV`
(`TrackerSettings.h:23`), т.е. это дефолтный решатель библиотеки.

---

### C3. `exit(-1)`/`assert(0)` в библиотеке

**Где:** `src/Tracker/BaseTracker.cpp:604-608`.

```cpp
if (dist < 0 || dist > maxPossibleCost)
{
    assert(0);
    exit(-1);
}
```

**Симптом:** библиотека завершает **весь процесс** при численно плохом cost.

**Когда срабатывает:**

* `DistMahalanobis` (см. H3) даёт отрицательный cost из-за `log(det(...))`
  → `exit(-1)` на первом же кадре;
* `DistLocalCartesian` + гомография с нулевым/отрицательным `w`
  (точка на линии горизонта) → `inf` → `exit(-1)`;
* любая будущая метрика, вернувшая `NaN` (NaN не отловится условием, но
  испортит решатель) или `inf`.

**Почему это критично:** трекер — библиотека; хост-приложение (сервер,
ROS-нода, Python-процесс) не должно умирать из-за одного плохого детекта.

**Исправление:** заменить на мягкую обработку:

```cpp
if (!std::isfinite(dist) || dist < 0)
{
    dist = maxPossibleCost;      // или пропустить пару
}
dist = std::min(dist, maxPossibleCost);
```

и, по желанию, `throw std::runtime_error` в отладочной сборке / счётчик
диагностики. `exit`/`assert` убрать полностью.

---

### C4. Выход за границы heat map

**Где:** `example/CarsCounting.cpp:442-456`.

```cpp
for (int y = 0; y < rect.height; ++y)
{
    float* heatPtr = m_heatMap.ptr<float>(rect.y + y) + rect.x;
    for (int x = 0; x < rect.width; ++x)
        heatPtr[x] += w;      // rect может выходить за кадр
}
```

**Симптом:** порча памяти (heap corruption) / падение при `--heat_map=1`,
если трек частично вышел за кадр: `rect.x < 0`, `rect.y < 0`,
`rect.x + rect.width > cols`, `rect.y + rect.height > rows`.
`TrackingObject::m_rrect` не гарантирует нахождение внутри кадра
(`CTrack` удаляет такие треки не мгновенно, а по таймауту/выходу за кадр).

**Исправление:**

```cpp
const cv::Rect r = rect & cv::Rect(0, 0, m_heatMap.cols, m_heatMap.rows);
for (int y = r.y; y < r.y + r.height; ++y)
{
    float* row = m_heatMap.ptr<float>(y);
    for (int x = r.x; x < r.x + r.width; ++x)
        row[x] += w;
}
```

**Проверка:** ASan-прогон примера с объектом, входящим в кадр с края,
`--heat_map=1`.

---

### C5. Некорректный ROI при вставке карты в кадр

**Где:** `example/CarsCounting.cpp:340-343`.

```cpp
cv::Size mapPreview(cvRound(frame.cols * k),
                    cvRound(((frame.cols * k) / geoMap.cols) * geoMap.rows));
cv::resize(geoMap, frame(cv::Rect(frame.cols - mapPreview.width - 1,
                                  frame.rows - mapPreview.height - 1,
                                  mapPreview.width, mapPreview.height)), ...);
```

**Симптом:** `cv::Exception` (assert `0 <= roi.x ...`) или запись мимо кадра,
если карта «портретная»/большая: `mapPreview.height > frame.rows` или
`mapPreview.width > frame.cols` → отрицательные `x/y` или ROI больше кадра.
Геопривязка заявлена как фича (`USE_GEOGRAPHICLIB`, `--geo_bind`).

**Исправление:** вычислить масштаб с учётом обеих сторон, проверить
пересечение и вставлять только при валидном ROI:

```cpp
const double s = std::min({0.25, 0.25 * frame.rows / geoMap.rows,
                           frame.cols / (double)geoMap.cols});
cv::Size sz(cvRound(geoMap.cols * s), cvRound(geoMap.rows * s));
if (sz.width < frame.cols && sz.height < frame.rows) { ... }
```

---

### C6. `ParseDFINE` / `ParseDFINE_IS` (OpenCV DNN)

**Где:** `src/Detector/OCVDNNDetector.cpp:1055-1093` и `1104`.

Проблемы:

1. `int64_t* labels = (int64_t*)detections[0].data;` — тип blob не проверяется.
   OpenCV DNN для целочисленных выходов ONNX часто отдаёт `CV_32S`
   (`int`), а не `int64`. Тогда на каждый элемент читается 8 байт из
   4-байтового буфера: мусорные class id **и чтение за границей** для
   последних элементов.
2. Порядок выходов жёстко зашит по индексам `[labels, boxes, scores]` и
   никак не проверяется. В TensorRT-ветке (`DFINE_bb.hpp:17-19`) порядок
   задаётся именами и другой — `[scores, labels, boxes]`. Если реальный
   порядок выходов ONNX/OpenCV отличается от ожидаемого, код молча
   интерпретирует один тензор как другой: классы/боксы будут мусорными без
   единого предупреждения. Расхождение двух реализаций — сигнал, что
   соответствие имён и индексов нужно проверять явно.
3. `ParseDFINE_IS` — `assert(0)`; в Release просто возвращает пустой список
   (пользователь видит «0 детекций»), в Debug — abort. При этом модель
   заявлена в README и в `ModelType::DFINE_IS`, а `net_type = DFINE_IS`
   присутствует в `data/settings_dfine_seg.ini`.

**Исправление:** выбирать выходы по именам из `m_outNames` (а не по индексу),
проверять `depth()`/`size()` перед интерпретацией, привести метки через
`cv::Mat::convertTo(CV_32S)`; реализовать `ParseDFINE_IS` (или убрать модель
из списка поддерживаемых и из INI).

---

### C7. TensorRT-инициализация: «успех» при ошибке, отсутствующие модели, `assert`

**Где:** `src/Detector/tensorrt_onnx/YoloONNX.cpp`,
`src/Detector/tensorrt_onnx/class_detector.cpp`,
`src/Detector/ONNXTensorRTDetector.cpp`,
`async_detector/AsyncDetector.cpp`.

1. **Ложный успех.** Если `deserializeCudaEngine` вернул `nullptr`,
   выставляется `res = true` (строки 95-104). Далее
   `m_buffers = make_unique<BufferManager>(m_engine=nullptr, 0)` (139) и
   `m_engine->createExecutionContext()` (141) — падение.
2. **Нет веток для YOLOV3/YOLOV3_TINY/YOLOV4/YOLOV4_TINY** в
   `YoloONNXImpl::Init` (55-126), хотя `ONNXTensorRTDetector::Init`
   принимает эти значения (81-119), пример `ONNXTensorRTExample` их
   предлагает (`example/examples.h:258-296`), а `async_detector` вообще
   не задаёт `net_type` → остаётся `YOLOV4` по умолчанию
   (`ONNXTensorRTDetector.cpp:14,28`). Итог: `m_detector` == `nullptr`,
   затем `m_detector->Init(m_params)` (152) — segfault. `async_detector`
   неработоспособен даже в сборке с TensorRT.
3. **`assert(res)` в `Detector::Init`** (class_detector.cpp:202) —
   Debug abort вместо кода ошибки; `ONNXTensorRTDetector::Init` вообще
   игнорирует возврат `Detector::Init` (163) и возвращает `true`, если
   `m_detector` не пуст (167).

**Статус:** ✅ исправлено (2026-11-11).

**Что сделано:**

* `YoloONNX::Init`: неудачная десериализация engine (а также пустой/нечитаемый
  файл, отсутствие runtime/engine/context/buffer) возвращает `false`;
  добавлена проверка ранга и размеров входного тензора.
* `GetBindings`: размерности тензоров собираются в порядке биндингов (на
  этот порядок рассчитаны парсеры моделей с несколькими выходами, например
  DFINE), при этом проверяется, что все объявленные входы и выходы
  присутствуют в engine.
* `YoloONNXImpl::Init`: явные ветки для YOLOV3/YOLOV3_TINY/YOLOV4/YOLOV4_TINY
  возвращают `false` с понятным сообщением (эти модели не поддерживаются
  ONNX/TensorRT-бэкендом); неизвестный `ModelType` также отклоняется;
  `m_detector` проверяется перед `Init`, при ошибке сбрасывается.
* `Detector::Init` больше не содержит `assert(res)` и возвращает реальный
  результат; `Detect`/`GetInputSize` безопасны после неудачной инициализации.
* `ONNXTensorRTDetector::Init` возвращает результат `Detector::Init` и
  сбрасывает обёртку при ошибке; `Detect` проверяет наличие детектора;
  `assert(0)` для неизвестного `net_type` и ошибок файла классов заменены на
  `return false` + лог. Тип по умолчанию изменён на `YOLOV8`.
* `async_detector`: конфиг переведён на `coco/yolo11s.onnx` с
  `net_type=YOLOV11`; при `nullptr`-детекторе конвейер останавливается, а не
  падает.

**Проверка:** собраны `yolo_rt_lib`, `mdetection`, `AsyncDetector` и
`MultitargetTracker` (MSVC Release). Запуск `-e=3` с `net_type=YOLOV4`
завершает инициализацию ошибкой `legacy YOLOv3/YOLOv4 models are not
supported...` без падения; запуск с реальным engine `yolo26m` (FP16)
печатает `YoloONNX::Init: successfully initialized` и обрабатывает кадры.

---

### C8. `async_detector`: гонки и `nullptr`

**Где:** `async_detector/AsyncDetector.cpp`, `async_detector/AsyncDetector.h`,
`async_detector/Queue.h`.

**Статус:** ✅ исправлено (2026-11-11).

**Что сделано:**

* `stopFlag` заменён на `std::atomic<bool>` во всех потоках
  (`CaptureThread`/`DetectThread`/`TrackingThread` и `Process`);
  `m_fps`/`m_framesCount` также сделаны атомарными, чтобы убрать гонки
  между потоком захвата и главным потоком.
* `FramesQueue::Signal` и `SetBreak` синхронизированы с `m_mutex`:
  раньше возможна была потерянная нотификация (проверка предиката и
  `wait()` не были упорядочены с изменением состояния), из-за чего
  потребитель мог заснуть навсегда. При ошибке открытия видео теперь
  вызывается `SetBreak(true)`.
* Каталог `logs/` создаётся через `fs::create_directories`; создание
  `rotating_file_sink_mt` обёрнуто в `try/catch`, при ошибке логгер
  продолжает работать только с консольным sink (без `std::terminate`).
* `nullptr`-детектор (ошибка инициализации) останавливает конвейер:
  выставляется `stopFlag` и `SetBreak`, поток детекции завершается.

**Проверка:** `AsyncDetector` собирается (MSVC Release). Запуск с
несуществующим видео завершается штатно (`Can't open ...`, `Correct exit`),
каталог `logs/` и лог-файл создаются. Если `logs` — обычный файл, выводятся
предупреждения о невозможности создать каталог/sink, приложение продолжает
работу и завершается. Запуск на `data/atrium.avi` с закэшированным engine
`yolo11s` обработал 10 кадров и завершился без зависаний (`Signal`/`SetBreak`
под нагрузкой).

---

### C9. Чтение за границей blob в YOLO-парсерах

**Где:** `src/Detector/OCVDNNDetector.cpp:586-589` (и аналоги: 642-645,
697-700, 796-799, 851-854, 901-904, 1274-1277, 1321-1324):

```cpp
cv::Mat scores(1, static_cast<int>(m_classNames.size()), CV_32FC1, classes_scores);
cv::minMaxLoc(scores, ...);
```

**Симптом:** если число классов в `classNames` больше, чем каналов в выходе
модели (например, `coco.names` из 91 строки для 80-классовой модели,
кастомный файл, или `classNames` вообще не задан и остались 21 VOC-класс),
`minMaxLoc` читает `classNames.size() * 4` байт из строки выхода. Для
последнего детекта это чтение за пределами буфера → мусорные confidence
и потенциальный segfault.

**Исправление:** проверять `dimensions - 4 >= (int)m_classNames.size()`
(для v8/v11), `dimensions - 5 >= ...` (v5) и либо ограничивать число
классов, либо логировать и использовать фактическое число классов из
формы выхода. Дополнительно проверять `detections.size()` (H19) и
`detections[0].dims`.

---

### C10. Батч TensorRT: OOB по `sampleImages`

**Где:** `src/Detector/tensorrt_onnx/YoloONNX.cpp:382-431`.

`inputB` берётся из формы входа engine (`.d[0]`), а цикл `for (b = 0; b <
inputB; ++b)` индексирует `sampleImages[b]` и `m_resizedBatch[b]`. Если
engine собран под батч 4, а пришёл 1 кадр (типичный случай для кропов —
`ONNXTensorRTDetector::Detect`, 229-236), — чтение за границей `std::vector`
`sampleImages`. Также `ProcessInputAspectRatio(frames)` с пустым `frames`
сразу обращается к `sampleImages[0]`.

**Исправление:** `const int batch = std::min(inputB, (int)sampleImages.size());`
и раньше — `if (frames.empty()) return false;`. Для корректной работы
переменного батча — `context->setInputShape(...)` под фактический размер
(или собирать engine под batch=1 / фиксированный batch и валидировать
вход).

---

### C11. Гистограммы по ROI без проверки границ

**Где:** `src/Tracker/BaseTracker.cpp` (`CTracker::CalcEmbeddins`),
`src/Tracker/track.cpp` (`CTrack::CalcDistHist`).

**Статус:** ✅ исправлено (2026-11-11).

**Симптом:** `cv::Exception` (assert в `Mat::Mat(const Mat&, const Rect&)`)
при детекции, выходящей за кадр или имеющей нулевой/отрицательный размер.
Включается **по умолчанию**: `TrackerSettings()` задаёт
`DistJaccard=0.5, DistHist=0.5` (`TrackerSettings.h:232-240`), и
`CalcEmbeddins` вызывает `calcHist`, если вес `DistHist > 0`.

Детекторы могут выдавать такие боксы: `left`/`top` могут быть отрицательными,
`width/height` — нулевыми (`ParseYOLOv10`, `ParseDFINE`, OBB и т.д.).
`CRegion` не клэмпит `m_brect` к кадру.

**Что сделано:**

* `CalcEmbeddins` проверяет, что кадр непустой, и пересекает `m_brect` с
  кадром (`roi = m_brect & frameRect`). Если `roi` меньше 2×2 или кадр
  пуст, гистограмма/эмбеддинг для региона не считаются.
* То же ограничение применяется к ReID-эмбеддингам, чтобы
  `EmbeddingsCalculator::Calc` не получил некорректный ROI.
* `CTrack::CalcDistHist` больше не вызывает `assert(0)`/`CV_Assert`:
  при отсутствии гистограммы (или несовпадении размера/типа, например
  при смене числа каналов кадра) возвращается максимальное расстояние `1`.
  Это позволяет трекеру продолжить работу, а не завершиться исключением.

**Проверка:** отдельный тест (`build/c11_test/test.cpp`) с
`TrackerSettings()` по умолчанию и кадром 160×120 подаёт боксы:
полностью вне кадра, частично вне кадра, вырожденный `0×0` и валидный.
Первый `Update` — 4 трека, второй `Update` с пустым кадром — без исключений,
`ok`. `mtracking`, `AsyncDetector` и `MultitargetTracker` собираются.

---

### C12. `maxTraceLength <= 0` → исключение

**Где:** `src/Tracker/track.cpp` (`CTrack::Update`, `CTrack::TrimTrace`).

**Статус:** ✅ исправлено (2026-11-11).

**Симптом:** при `max_trace_len = 0` (или отрицательном) из INI условие
`period > maxTraceLength` истинно всегда → `pop_front` очищает траекторию →
следующий `m_trace.at(0)` бросает `std::out_of_range`, которое никто не
ловит → `std::terminate`. `max_trace_len` — публичная настройка
(`TrackerSettings.h:80`, `data/settings*.ini`), значение 0 выглядит
допустимым («не хранить траекторию»).

**Что сделано:**

* обрезка истории вынесена в `CTrack::TrimTrace`;
* цикл обрезки проверяет `m_trace.size() > 0` и не вызывает `at(0)` на
  пустом векторе;
* при `maxTraceLength <= 0` хранится только последняя точка, а не пустая
  траектория: потребители (`DrawTrack`, `GetTrajectory`) рассчитывают на
  `size() >= 1` (в циклах используется `size() - 1`);
* если история всё же оказалась пустой, последняя точка восстанавливается
  защитно.

**Проверка:** тест `build/c12_test/test.cpp` с `maxTraceLength = 0, -1,
0.001, 2.0`: 10 кадров с детекциями и 5 кадров без детекций; ни одного
пустого трека и ни одного исключения, `failures=0`. `mtracking` и
приложения собираются.

---

### C13. VIBE: гонка RNG и `% 0`

**Где:** `src/Detector/vibe_src/vibe.cpp`, `src/Detector/vibe_src/vibe.hpp`.

**Статус:** ✅ исправлено (2026-11-11).

* `m_rng`/`m_rngIdx` — общие поля, изменяемые в `#pragma omp parallel for`
  (строки 84, 117, 119, 126-127) без синхронизации → гонка данных (UB) и
  «порча» последовательности; при нескольких OpenMP-потоках возможно
  чтение/запись одного индекса.
* `m_updateFactor` приходит из конфига (`updateFactor`, `MotionDetector`).
  `% m_updateFactor` (117) при 0 → деление на 0 (SIGFPE/UB). Аналогично
  `% m_samples`, `% neighbor_count` (27) при нулевых значениях.

**Что сделано:**

* общий `m_rng`/`m_rngIdx` удалён; используются thread-local
  `std::mt19937` и функции `RandomIndex`/`RandomHit` — состояние ГПСЧ
  больше не разделяется потоками OpenMP;
* конструктор `VIBE` валидирует параметры: `samples >= 1`, `channels >= 1`,
  `pixelNeighbor >= 0`, `distanceThreshold >= 1`, `matchingThreshold >= 0`,
  `updateFactor >= 1`; деление по модулю на 0 исключено;
* `getRndNeighbor` проверяет `neighbor_count` перед выбором соседа;
* `update` использует `RandomHit`/`RandomIndex` вместо прямого `%`.

**Проверка:** тест `build/c13_test/test.cpp` создаёт VIBE с
`samples=0, pixelNeighbor=-5, updateFactor=0` (нулевые/отрицательные
параметры), прогоняет нормальную модель 50 кадров и неконтинуальный ROI:
маски 64×64, 128×96, 64×64, `ok`; `mdetection` собирается.

---

### C14. Python-обёртка не собирается и отдаёт висячие указатели

**Где:** `src/python_bind/mtracker.cpp`.

1. `#include "../Tracker/Ctracker.h"` (11) — файла `Ctracker.h` в репозитории
   нет (есть `BaseTracker.h`), а `README.md:112` тоже ссылается на
   несуществующий заголовок. Сборка `MTRACKER_PYTHON=ON` падает на
   препроцессоре.
2. `PyBaseTracker::UpdateMat(const regions_t&, cv::Mat, float fps) override`
   (256-259) — в `BaseTracker` нет такого виртуального метода (есть
   `UpdateMat(..., time_point_t)`), `override` не компилируется.
3. `mdetector.def("Detect", &MotionDetector::Detect)` (393) — `Detect`
   принимает `const cv::UMat&`, а кастера `cv::UMat` в модуле нет →
   ошибка компиляции pybind11.
4. `type_caster<cv::Mat>::cast` (112) возвращает `py::array` поверх
   `mat.data` без копии/владения: `cv::Mat` — локальная переменная
   `cast`, после возврата данные освобождаются, numpy-массив остаётся
   висячим (`read_image`, `passthru`, `clone`).
5. `BaseTracker::CreateTracker` имеет два аргумента (settings, fps), а
   `demo.py:73` вызывает `mt.MTracker(tracker_settings)`; `Update` в
   demo (`demo.py:86`) передаёт `fps` float вместо `time_point_t`.
   То есть даже после починки компиляции demo.py не заработает.

**Исправление:** переписать модуль под актуальный API, добавить кастер
`cv::UMat` (или перейти на `cv::Mat` в биндингах), возвращать
`py::array` с копией данных (или capsule с `Mat`), обновить `demo.py`,
добавить CI-джобу сборки модуля и smoke-тест.

---

## 4. P1 — подробно

### H1. Нормализация при отсутствии эмбеддинга

**Где:** `src/Tracker/BaseTracker.cpp:552-570`.

```cpp
if (resCos.second)  dist += weight * resCos.first;
else                dist /= weight;          // ← неверно
```

Если косинусное расстояние посчитать нельзя (нет модели для этого типа,
пустой эмбеддинг), код **делит уже накопленную** сумму на вес косинуса.
При `DistFeatureCos = 1` (единственная метрика) `dist` становится 0 для
любой пары → все пары «идеальны» → произвольные назначения и ложные
треки. При смешанных весах искажаются вклады остальных метрик.

**Исправление:** не добавлять вклад и не нормировать; либо нормировать на
сумму **оставшихся** весов:

```cpp
const track_t restWeight = 1.0f - m_settings.m_distType[tracking::DistFeatureCos];
if (restWeight > 1e-6f) dist /= restWeight;
```

и документировать, что при отсутствии эмбеддинга метрика исключается.

---

### H2. `GetVelocity` и индексы состояния

**Где:** `src/Tracker/Kalman.cpp:1206-1244`.

```cpp
int indX = 2, indY = 3;
if (statePre.rows > 4) { indX = 4; indY = 5; }
```

Фактические раскладки состояний (`Kalman.cpp:50-54, 173-183, 247-253,
307-319`):

| Модель | state | vx, vy | Что вернёт текущий код |
|---|---|---|---|
| point, const v | 4 | 2, 3 | 2, 3 ✓ |
| point, const a | 6 | 2, 3 | 4, 5 = ax, ay ✗ |
| rect, const v | 8 | 4, 5 | 4, 5 ✓ |
| rrect, const v | 10 | 5, 6 | 4, 5 = angle, vx ✗ |
| rect, const a | 12 | 4, 5 | 4, 5 ✓ |

**Последствия:** неверная скорость в `TrackingObject::m_velocity`,
неверный эллипс предсказания (`CalcPredictionEllipse`), неверный
`CheckStatic` для `FilterRRect`/`useAcceleration`, неверная отрисовка
вектора скорости.

**Исправление:** выбирать индексы по числу состояний и включённому
ускорению (лучше — хранить метаданные о модели в `TKalmanFilter`).

---

### H3. Mahalanobis-метрика

**Где:** `src/Tracker/track.cpp:184-205`, `src/Tracker/Kalman.cpp:1247-1270`.

1. В качестве ковариации берётся `processNoiseCov` («шум процесса»), а не
   инновационная ковариация `S = H·P·Hᵀ + R`. Это математически неверно:
   расстояние не отражает неопределённость фильтра.
2. `measurePoint` всегда 4×1, а `predictPoint = statePost` имеет 4/6/8/10/12
   строк. Для `FilterRect` (8/12) и `FilterRRect` (10) `cv::Mahalanobis`
   бросит `cv::Exception` из-за несовпадения размеров. Проверка
   `predictPoint.rows == 2` (197) никогда не истинна.
3. `mahaDist += log(determinant(res1))` — для `processNoiseCov` определитель
   равен нулю (матрица шума процесса вырождена по построению: `n2² = n1·n3`),
   поэтому `log(0) = -inf`, cost становится `-inf` → см. C3 (`exit(-1)`).

**Исправление:** реализовать `H`, `P`, `R`, вычислить `S` и расстояние
`sqrt((z - Hx)ᵀ S⁻¹ (z - Hx))` для согласованной размерности; не добавлять
`log det` (или добавлять с ограничением и только как диагностику).

---

### H4. Единицы угла в эллипсе предсказания

**Где:** `src/Tracker/track.cpp:439-483`.

```cpp
rrect.angle = atanf(d.y / d.x);        // радианы (445-463)
...
track_t t = acosf(pt_.x / r);          // радианы
track_t t_ = t - rrect.angle;          // радианы - радианы? Нет: angle уже в RotatedRect
```

`cv::RotatedRect::angle` — **градусы** (OpenCV), а `atanf` возвращает
радианы. При этом `IsInsideArea` вычисляет полярный угол `t` в радианах и
вычитает `rrect.angle` как есть. В результате для движения под углом
(кроме 0°/90°) эллипс повёрнут неверно, и `DistCenters`/`DistRects` дают
искажённые расстояния. `GetLastRect` также возвращает этот `RotatedRect`.

**Исправление:** хранить угол в градусах (`rrect.angle = atan2f(d.y,
d.x) * 180.f / CV_PI`), а в `IsInsideArea` переводить:
`const track_t angleRad = rrect.angle * CV_PI / 180.f;`.

---

### H5. Inertia correction на потерянных кадрах

**Где:** `src/Tracker/Kalman.cpp:769-782` (point), `948-963` (rect).

`CTrack::PointUpdate`/`RectUpdate` при `dataCorrect == false` передают
пустой `CRegion()` c центром (0,0) и нулевым rect. Внутри `Update`
`measurement` корректно берётся из `m_lastPointResult`, но «inertia»
считает `currDist` между `estimated` и **переданным** `pt`/`rect`:

```cpp
track_t currDist = sqrtf(sqr(estimated.at(0) - pt.x) + ...);
if (currDist > m_lastDist) m_deltaTime = min(m_deltaTime + step, max);
```

То есть на каждом кадре без детекции `currDist` — расстояние до (0,0) →
`m_deltaTime` уходит в максимум. Это меняет `transitionMatrix` и динамику
фильтра именно тогда, когда трек потерян.

**Исправление:** вычислять `currDist` только при `dataCorrect` (или
передавать реальную последнюю позицию/предсказание).

---

### H6. Матрица ускорения для rect-модели

**Где:** `src/Tracker/Kalman.cpp:307-319`.

Состояние: `[x, y, w, h, vx, vy, vw, vh, ax, ay, aw, ah]`.
Строка 0 матрицы перехода:

```cpp
{ 1, 0, 0, 0, dt, 0, 0, 0, dt2, 0, dt2, 0 }
```

`x' = x + dt·vx + dt2·ax + dt2·aw` — ускорение ширины `aw` попадает в
координату `x`; строка 1 аналогично подмешивает `ah` в `y`. Для `w`/`h`
строки корректны. При `useAcceleration = true` и `FilterRect` модель
физически неверна.

**Исправление:** строка 0 — `dt2` только в столбце `ax` (8), строка 1 —
только в `ay` (9); остальные — по аналогии. Добавить тест: при нулевых
скоростях и `ax > 0` `w` не меняется.

---

### H7. Неподдерживаемые комбинации Kalman/FilterGoal

**Где:** `Kalman.cpp:383-387` (`CreateLinearAcceleration(RotatedRect)` —
`assert(0)`), `1090-1105` (UKF + FilterRRect — `assert(0)`),
`812-838` (`GetRectPrediction` для UKF читает `prediction(2),(3)` как
ширину/высоту, хотя в UKF-состоянии это `vx, vy`).

**Последствия:**

* `FilterRRect + useAcceleration`: в Release фильтр остаётся
  неинициализированным (трек «не фильтруется»), в Debug — abort;
* `UKF + FilterRRect`: то же;
* `UKF + FilterRect`: предсказание портит `m_lastRectResult`, и на
  потерянных кадрах measurement становится мусорным.

**Исправление:** либо реализовать модели, либо валидировать настройки в
`CTracker`/`CreateTracker` и возвращать ошибку (не `assert`), либо
документированно понижать до `KalmanLinear`.

---

### H8. ByteTrack: статика, память, API

**Где:** `src/Tracker/byte_track/BYTETracker.cpp`.

```cpp
std::chrono::duration<double> period = m_lastFrameTime - m_lastFrameTime; // всегда 0
...
removed_stracks_ = jointStracks(removed_stracks_, current_removed_stracks); // растёт вечно
```

* `period` всегда 0 → `m_staticTime` всегда 0; статичность ByteTrack не
  поддерживает (`isStatic=false`).
* `removed_stracks_` не очищается (в оригинальном ByteTrack очищается
  каждый кадр) → неограниченный рост памяти на длинном видео; при этом
  старые ID там уже не нужны.
* `GetRemovedTracks` (43-51) **добавляет** в переданный вектор, не очищая
  его, в отличие от `CTracker::GetRemovedTracks` (assign). Повторные
  вызовы дают дубликаты.

**Исправление:** считать `period` от времени появления/последней
статичности; очищать/тримить `removed_stracks_`; `trackIDs.clear()` в
начале `GetRemovedTracks` (или задокументировать контракт и привести
`CTracker` к тому же).

---

### H9. Статичность: секунды как кадры, скорость не учитывается

**Где:** `src/Tracker/track.cpp:516-580`, `src/Tracker/BaseTracker.cpp:415`.

`CheckStatic(int trajLen, ...)` вызывается с `trajLen = m_minStaticTime`
(по документации — **секунды**, `TrackerSettings.h:88-92`), а используется
как **количество точек траектории**:

```cpp
if (!trajLen || static_cast<int>(m_trace.size()) < trajLen) ...
cv::Rect centerROI(... m_trace[m_trace.size() - trajLen] ...);
```

При 30 fps «5 секунд» проверяются по 5 кадрам. Плюс условие
`speed < maxSpeedForStatic` закомментировано (541), а переменная `speed`
не используется — настройка `m_maxSpeedForStatic` (и ключ
`max_speed_for_static`) не влияет ни на что. `m_staticFrame` клонируется,
но нигде не читается.

**Исправление:** либо переименовать параметр в `minStaticFrames` и
пересчитывать секунды в кадры по fps, либо сравнивать время
(`currTime - m_staticStartTime`); вернуть проверку скорости (в единицах
метрики) или удалить настройку.

---

### H10. CarsCounting игнорирует `[tracking]`

**Где:** `example/CarsCounting.cpp:247-284`.

`m_trackerSettings` читается из INI (`52`), но затем создаётся локальный
`TrackerSettings settings;`, заполняется «хардкодом» и передаётся в
`CreateTracker`. Все настройки трекера из `--settings` (дистанция, dt,
порог, потеря, статика, ByteTrack и т.д.) молча игнорируются — в отличие
от детектора, где используется `m_trackerSettings`.

**Исправление:** использовать `m_trackerSettings` (как в
`OpenCVDNNExample::InitTracker`), переопределяя только те поля, которые
действительно нужны примеру, и/или добавить отдельный раздел
`[cars_counting]`.

---

### H11. Геопривязка: гонки и деление на ноль

**Где:** `example/CarsCounting.cpp:68-72, 241-245`,
`example/CarsCounting.h:396-410, 507-521`.

* `m_geoBindingsRead` — обычный `bool`, читается/пишется из
  `Detection` (поток захвата/детекции в async-режиме) и из `InitTracker`
  (главный поток) → гонка и возможная двойная инициализация
  `m_geoParams` параллельно с использованием.
* `Pix2Ground`/`Ground2Pix` делят на `g[2]`/`p[2]` без проверки: для
  точки на линии горизонта (или за камерой) знаменатель 0/отрицателен →
  `inf`/`NaN`/огромные координаты. Дальше это уходит в
  `DistLocalCartesian` → cost → C3.
* `Pix2MapParams::Init` при неудаче `GetPerspectiveTransform` оставляет
  `m_toPix` неинициализированной (нет флага валидности), а
  `DrawTracksOnMap` её использует.
* Нет проверки качества гомографии: коллинеарные/совпадающие точки,
  дубликаты, точки вне кадра принимаются молча.

**Исправление:** `std::once_flag`/мьютекс для `ReadGeobindings`;
`if (std::abs(w) < 1e-9) return {-1,-1}`/пропуск региона; флаг
`m_valid` для `Pix2MapParams`; проверка reprojection error и
невырожденности при `SetKeyPoints`.

---

### H12. `RoadLine`: арифметика и неинициализированное значение

**Где:** `example/CarsCounting.h:590-735`.

```cpp
pt.y = (m_pt1.y + m_pt1.y) / 2.f - 0.01f;   // 4 раза, должно быть m_pt1.y + m_pt2.y
```

Из-за опечатки средняя точка линии считается неверно → неверно
определяется направление пересечения (`intersect1`/`intersect2`), т.е.
подсчёт «въехало/выехало» врёт. Кроме того, в `CheckIntersection`
`intersectPt.x` не инициализируется, если обе линии вертикальны или
параллельны (`trivial1 && trivial2`, `fabs(a2-a1) <= eps`), а затем
читается (`intersectPt.y = a1 * intersectPt.x + b1`, `InRange(...)`) —
UB/мусор.

**Исправление:** заменить `m_pt1.y + m_pt1.y` на `m_pt1.y + m_pt2.y`;
инициализировать `intersectPt` (например, `cv::Point2f(0,0)`) и выходить
из функции при параллельности/совпадении прямых; добавить unit-тесты
пересечений (в т.ч. вертикальные/горизонтальные линии).

---

### H13. `Esc` не выходит из видео в `SyncProcess`

**Где:** `example/VideoExample.cpp:220-241`.

`break` при `k == 27` находится внутри `for (i = 0; i < m_batchSize; ++i)`
и прерывает только батч; внешний `for (;;)` продолжает читать видео.
То же для `AsyncProcess` (309-339) — там `key == 27` проверяется после
внутреннего цикла, но внутренний `else break` ломает обработку батча
(см. M18); для `batch_size == 1` Esc работает.

**Исправление:** выставить флаг `stop`/`return` из внешнего цикла, а
внутренний `break` дополнить проверкой `if (key == 27) return;`.

---

### H14. Неверный thread-guard в `Frame`

**Где:** `example/VideoExample.h:119-135`.

```cpp
if (m_umBGR.empty() || !m_umBGRGenerated || lastThreadID != m_umGrayThreadID)
```

Должно быть `m_umBGRThreadID`. Из-за опечатки решение «пересоздавать ли
UMat» опирается на идентификатор потока, работавшего с **серым** UMat.
В двухпоточном конвейере это может привести к использованию UMat,
созданного в другом потоке (проблемы с OpenCL-контекстом) или к лишним
конвертациям.

**Исправление:** заменить поле; добавить тест/ассерт, что
`m_umBGRThreadID` обновляется при создании BGR-UMat.

---

### H15. Неинициализированное `m_staticStartTime`

**Где:** `src/Tracker/track.cpp:616-623`, `src/Tracker/track.h:196`.

`m_staticStartTime` — `time_point_t` по умолчанию (эпоха). До первого
`CheckStatic` (первый `Update` с детекцией) `ConstructObject` считает
`period = frameTime - m_staticStartTime` ≈ 1.7·10⁹ секунд. Для нового
трека, который ещё не обновлялся (создан в текущем кадре), `m_staticTime`
огромен. Поле публичное (`TrackingObject::m_staticTime`) и доступно из
Python/логов.

**Исправление:** инициализировать `m_staticStartTime(currTime)` в
конструкторе `CTrack`; в `ConstructObject` возвращать 0, если
`!m_isStatic` (или отдельное поле «время с последней детекции»).

---

### H16. Python-упаковка и demo.py

**Где:** `setup.py`, `demo.py`.

* `from distutils.command.install_data import install_data` и
  `from distutils.version import LooseVersion` (8, 9) — `distutils` удалён
  в Python 3.12 → установка падает.
* `os.makedirs(build_dir)` без `exist_ok=True` (138) → повторная сборка
  падает; `pyd_path = pyd_path[0]` (176) → `IndexError`, если модуль не
  собрался; копируется только один файл.
* `-DBUILD_YOLO_LIB`/`-DBUILD_YOLO_TENSORRT` (151-152) — таких опций
  больше нет (есть `BUILD_ONNX_TENSORRT`).
* `demo.py` вызывает `mt.MTracker(tracker_settings)` без `fps` и
  `mtracker.Update(regions, img, fps)` с `float` вместо `time_point_t`
  — несовместимо с текущим C++ API (см. C14).

**Исправление:** перейти на `setuptools`/`pyproject.toml`, `exist_ok`,
проверять результат CMake, обновить demo.py, версионировать из одного
источника. Добавить CI-джобу «wheel + `python -c "import pymtracking"`».

---

### H17. INI с ошибкой считается валидным

**Где:** `src/Tracker/TrackerSettings.cpp:16`.

```cpp
if (reader.ParseError() >= 0)
```

`INIReader::ParseError()` возвращает -1 (файл не найден), 0 (успех) или
номер строки с ошибкой (> 0). Условие `>= 0` принимает и файл с ошибкой:
часть настроек будет прочитана, часть — нет, а вызывающий код получит
`true` («settings loaded») и не узнает о проблеме.

**Исправление:** `if (reader.ParseError() == 0)`; для `> 0` — лог с
номером строки и `return false` (или явный partial-режим).

---

### H18. `GetCrops` может зависнуть

**Где:** `src/Detector/BaseDetector.h:229-277`.

```cpp
int cropHeight = cvRound(maxCropRatio * netSize.height);
...
const int stepX = 3 * cropWidth / 4;
const int stepY = 3 * cropHeight / 4;
for (int y = 0; y < imgSize.height; y += stepY) ...
```

Если `maxCropRatio` мал (например, `0.001`) или `netSize` мал, `cropWidth`
/`cropHeight` могут округлиться до 0 → `stepX/stepY == 0` → бесконечный
цикл (и деление на 0 в `netSize.width / netSize.height`, если размер 0).
`maxCropRatio` берётся из INI/конфига без валидации.

**Исправление:** `if (cropWidth < 1 || cropHeight < 1 || netSize.width <= 0
|| netSize.height <= 0) return {};` и `stepX = std::max(1, ...)`.

---

### H19. Выходы сети без проверки количества

**Где:** `OCVDNNDetector.cpp:558-665, 927-1047, 1055-1093`.

`ParseYOLOv5`/`8`/`9`/`10`/`11`, `ParseRFDETR`, `ParseDFINE` обращаются к
`detections[0]`, `detections[1]`, `detections[2]` без проверки
`detections.size()`. Если модель вернула меньше выходов (несовпадение
версии/экспорта, ошибка `forward`), это OOB `std::vector` — UB, обычно
падение. Сигнатуры `Parse*` принимают `const std::vector<cv::Mat>&` без
размера.

**Исправление:** в `DetectInCrop` проверить `detections.size()` до вызова
парсера, передавать ожидаемое число/имена выходов; внутри парсеров —
проверять `dims`, `size`.

---

### H20. `DistType` в `Update` почти ничего не значит

**Где:** `src/Tracker/BaseTracker.cpp:119-137`.

```cpp
void CTracker::Update(..., tracking::DistType distType)
{
    UpdateTrackingState(..., distType == tracking::DistLocalCartesian);
}
```

Параметр используется только как булев флаг «ENU-режим». Реальный выбор
метрик — веса `m_distType` в настройках. Пользователь, вызвавший
`Update(..., DistCenters)`, ожидает смену метрики, но получит прежнюю.
Плюс в перегрузке без `distType` захардкожен `DistJaccard` (только как
флаг).

**Исправление:** либо убрать параметр, либо действительно переключать
веса (`SetDistance`-семантика) и документировать.

---

### H21. Небезопасные/неожиданные дефолты `TrackerSettings`

**Где:** `src/Tracker/TrackerSettings.h:23,173,232-240`.

* `m_matchType = MatchLAPJV` — дефолтный решатель затронут C2;
* `m_detectorBackend = 5` (TensorRT) — «сырой» `TrackerSettings` по
  умолчанию требует TensorRT, которого в обычной сборке нет.
  `ParseTrackerSettings` при отсутствии ключа подставляет `DNN_OCV`
  (`TrackerSettings.cpp:78`), поэтому проблема проявляется в коде, который
  использует структуру без INI/явного выбора бэкенда;
* `DistHist = 0.5` вместе с `DistJaccard = 0.5` включает гистограммы по
  умолчанию → C11 при боксах вне кадра.

**Исправление:** `MatchHungrian` по умолчанию (или исправить C2 и покрыть
тестами), `m_detectorBackend = DNN_OCV` (или сделать обязательным),
`DistHist = 0` по умолчанию / включать только при наличии кадра и
валидного ROI.

---

### H22. Глобальный реестр типов

**Где:** `src/mtracking/object_types.h:14-33`.

* `Type2Str(type)` индексирует `m_typeNames[(size_t)type]` без проверки —
  любой `objtype_t >= size` (например, из внешнего API/битого файла)
  даёт OOB чтение.
* `Str2Type(name)` **добавляет** неизвестное имя в глобальный статический
  вектор и возвращает новый индекс. Из-за этого:
  * `FillTypesMap` всегда возвращает `true` (`Str2Type` не возвращает
    `bad_type`) → проверки «Unknown types in class names» и `assert(0)` —
    мёртвый код;
  * ID типов зависят от порядка инициализации детекторов/файлов;
  * нет потокобезопасности (вызовы из детекторов и примеров).
* `Type2Str`/`Str2Type` используются в горячем пути отрисовки/логов.

**Исправление:** проверять границы; сделать реестр явным (инициализация
из файла, неизменяемая после старта) или `thread_local`/мьютекс; для
неизвестных имён возвращать `bad_type`.

---

## 5. P2/P3 — что делать по группам

### 5.1 Производительность

* Убрать копирование кадра в `m_prevFrame` (M1) и клонирование
  `m_staticFrame` (M2) — это полные копии кадра на каждую итерацию/объект.
* `GetTracks` копирует траектории (M3). Для UI/логов достаточно
  `TrackingObject` без полной истории или с `shared_ptr` на неизменяемую
  траекторию.
* `CalcEmbeddins`: считать гистограммы/эмбеддинги только для реально
  используемых метрик и регионов; переиспользовать буферы.
* Заменить `std::find` (M4) на `std::vector<bool>`/битовую маску
  назначений.
* OpenMP-цикл (M5): либо убрать параллелизм для визуальных трекеров
  OpenCV, либо сериализовать создание/обновление `cv::Tracker`, либо
  использовать пул потоков с гарантией thread-safety OpenCV.

### 5.2 Надёжность и численные методы

* Валидировать все входные `CRegion` (клэмпить `m_brect` к кадру,
  отбрасывать `width <= 0`/`height <= 0`) на входе `CTracker::Update` —
  это закрывает C11, часть C9 и NaN-ы.
* Проверять `std::isfinite` для всех cost/координат; ввести единый
  «санитайзер» матрицы стоимостей.
* Валидировать `max_trace_len`, `max_lost_time`, `dt`, `min_static_time`,
  `minAreaRadius*` при парсинге настроек.
* `BackgroundSubtract::Init` должен возвращать `false`/код, если
  запрошенный алгоритм недоступен (M12).
* Обернуть парсинг INI/KeyVal в `try/catch` и давать понятные сообщения
  (M13).

### 5.3 API и архитектура

* Разделить `TrackerSettings` на `TrackerSettings` и `DetectorSettings`
  (или вложенные структуры), оставив совместимый парсер.
* Привести контракт `GetTracks`/`GetRemovedTracks` к единому виду (M24):
  `clear()` в начале у обоих трекеров, `const`, возврат по значению.
* Отделить публичные заголовки (`BaseTracker.h`, `TrackerSettings.h`,
  `trajectory.h`, типы) от внутренних (`track.h`, `Kalman.h`,
  `ShortPathCalculator.h`); обновить `PUBLIC_HEADER`/install.
* Убрать `exit`/`assert` из библиотечных путей (C3, H7), заменить на
  исключения/`expected`-подобный результат.
* Сделать `TypeConverter` неизменяемым после инициализации (H22).
* Ввести `DetectorFactory`-регистрацию вместо `switch` (упрощает
  добавление моделей и тестирование).
* Для TensorRT: валидировать имена/типы/размеры тензоров, не полагаться
  на порядок выходов; собирать engine под фактический batch или
  поддерживать динамический профиль (C7, C10, H19).
* Привести `m_velocity` к задокументированным единицам (M25) или
  переименовать в `m_velocityPerStep`; то же для `m_staticTime` (M26).

### 5.4 Сборка и упаковка

* Убрать переопределение `CMAKE_BINARY_DIR`, `unset(... CACHE)`;
  использовать `CMAKE_RUNTIME_OUTPUT_DIRECTORY`/`CMAKE_LIBRARY_OUTPUT_DIRECTORY`
  (M20).
* Вынести `-march=native`/`-funroll-loops` в отдельную опцию
  (`MT_NATIVE=ON/OFF`), Release по умолчанию — переносимый (M20).
* Единый тип библиотек на всех платформах (`SHARED`/`STATIC` по опции) и
  линковка OpenCV на MSVC (M21).
* `find_package(Threads REQUIRED)` + `Threads::Threads`; OpenMP — через
  `OpenMP::OpenMP_CXX` (M22).
* Удалить мёртвые include-пути/опции (M23).
* CMake-пакет: `MTTrackingConfig.cmake.in` не находит OpenCV/зависимости
  (`find_dependency(OpenCV)`), `LIB_DIR` вычисляется от
  `INCLUDE_INSTALL_DIR` — проверить и исправить.
* Python: `pyproject.toml`, `setuptools>=64`, CI-сборка wheel (H16, C14).
* Версия — из одного места (M36).

### 5.5 Документация

* README: актуальный include (`BaseTracker.h`), актуальные опции сборки,
  пример Python, список поддерживаемых моделей и ограничений (M37).
* `data/settings*.ini`: убрать опечатки/устаревшие значения, не хранить
  абсолютные пути (M38, M39).
* Добавить `docs/` (этот файл и `architecture.md`) в README.
* Задокументировать единицы величин (dt, velocity, static time,
  minAreaRadius*, maxTraceLength).

---

## 6. План работ (roadmap по этапам)

### M0. Стабилизация (1–2 недели)

Цель: убрать падения и UB; выпустить патч-релиз `1.1.1`.

- [ ] C1 — проверка `nullptr` в `CreateDetector`.
- [ ] C3 — убрать `exit(-1)`/`assert(0)` из `CreateDistaceMatrix`,
      заменить на soft-fail/исключение.
- [ ] C12 — защита цикла обрезки траектории.
- [ ] C11 — клэмпинг ROI/валидация `CRegion` перед гистограммами.
- [ ] C4, C5 — клэмпинг heat map и ROI карты.
- [ ] C13 — RNG в VIBE (thread-local) + валидация параметров.
- [ ] C8 — `std::atomic<bool>` в `async_detector`, проверка `nullptr`,
      создание `logs/`.
- [ ] H18 — защита от нулевых кропов.
- [ ] H13, H15 — быстрые поведенческие фиксы.
- [ ] Добавить ASan/UBSan-сборку в CI (хотя бы один workflow).

**Критерий готовности:** примеры `-e=1`, `-e=2`, `-e=4`, `async_detector`
запускаются под ASan на `data/atrium.avi` без ошибок; `Detectors`
недоступных бэкендов не роняют процесс.

### M1. Корректность трекинга и детекции (2–4 недели)

Цель: `1.2.0` с исправленными метриками и поддержкой заявленных моделей.

- [ ] C2 — исправить LAPJV + property-тесты против Hungarian.
- [ ] H1 — нормализация cosine.
- [ ] H2, H4, H5, H6 — Kalman/эллипс/inertia/матрица ускорения.
- [ ] H3 — Mahalanobis (или временно запретить комбинацию с валидацией).
- [ ] H7 — валидация `KalmanType`×`FilterGoal` (без `assert`).
- [ ] H9 — статичность в секундах + скорость.
- [ ] H8 — ByteTrack (память, `period`, контракт).
- [ ] C6, C9, C10, H19 — парсеры DNN/TensorRT: имена/типы/размеры.
- [ ] C7 — TensorRT init, ветки YOLOV3/V4, ошибки без `assert`.
- [ ] H11, H12 — геопривязка и линии.
- [ ] H10 — CarsCounting использует настройки.

**Критерий готовности:** регрессионные тесты на синтетических
последовательностях (пересечения, потеря, вход/выход из кадра,
OBB-модели); метрики MOTA/IDF1 не хуже текущих на `data/atrium.avi`.

### M2. Тесты, CI, воспроизводимость (параллельно M1)

- [ ] Юнит-тесты: LAPJV/Hungarian (включая прямоугольные и вырожденные
      матрицы), Kalman (модели/размерности), GEO (ENU ↔ WGS84, гомография),
      `nms`, `CRegion`, `TypeConverter`, INI-парсер.
- [ ] Интеграционные: синтетическое видео с известными траекториями
      (2–3 сценария), проверка ID-switch, статики, подсчёта линий.
- [ ] CI-матрица: Linux GCC/Clang, Windows MSVC, macOS; Debug+Release;
      ASan/UBSan; `MTRACKER_PYTHON=ON`; `BUILD_ASYNC_DETECTOR=ON`;
      `BUILD_CARS_COUNTING=ON`; `USE_OCV_UKF=ON`; опционально TensorRT
      (self-hosted runner).
- [ ] Бенчмарк: 1080p, 30 fps, 50 объектов — бюджет кадра.

### M3. Архитектурный рефакторинг (1–2 месяца)

- [ ] Разделить конфиги трекера/детектора; версионировать схему INI.
- [ ] Публичный API: стабильные заголовки, `namespace`, `MT_API`,
      семантическое версионирование; убрать внутренние типы из install.
- [ ] Вынести общие утилиты (логгер, `Frame`, тайминги) в отдельный
      модуль, убрать дублирование между `example` и `async_detector`.
- [ ] Реестр детекторов/трекеров вместо `switch`.
- [ ] TensorRT-слой: абстракция над версиями TensorRT, вынести
      `common/` из репозитория или зафиксировать как submodule.
- [ ] Python: pybind11 v2.11+, numpy-владение, `pyproject.toml`,
      типизация/докстринги.

### M4. Функциональное развитие (после стабилизации)

- [ ] Сегментационные маски для OpenCV DNN (сейчас только TensorRT).
- [ ] Полноценная поддержка `FilterRRect` (включая acceleration/UKF) или
      явный отказ.
- [ ] ReID: несколько моделей/типов, батч-инференс, кэширование.
- [ ] Метрики MOT (MOTA/IDF1/ID-switch) как отдельный инструмент.
- [ ] Экспорт треков в MOTChallenge/CVAT (сейчас CVAT есть, MOT — нет).
- [ ] GPU-декодирование/препроцессинг (опционально).

---

## 7. План тестирования (минимальный)

| Уровень | Что проверять | Инструмент |
|---|---|---|
| Unit | `lap()` vs эталон, `AssignmentProblemSolver`, `TKalmanFilter` (все модели), `CRegion` (вырожденные), `nms`, `TypeConverter`, `ParseTrackerSettings` | GoogleTest/Catch2 + CTest |
| Property | LAPJV: сумма cost ≤ Hungarian; перестановочность; прямоугольные матрицы | генератор случайных матриц |
| Sanitizers | ASan/UBSan на примерах и unit-тестах | CI |
| Integration | Синтетическое видео: 5 объектов, пересечения, потеря, статика; проверка ID и counts | OpenCV + эталонные CSV |
| Regression | `data/atrium.avi`, `data/TrackingBugs.mp4` — метрики и число треков | скрипт сравнения |
| Fuzz | Парсеры выходов DNN (shape/тип), INI-парсер, `str2ID` | libFuzzer/AFL (опционально) |
| Python | `import pymtracking`, `MTracker(settings, fps)`, `Update`, `GetTracks`, numpy-массивы | pytest |

---

## 8. Приложение A. Эскизы исправлений

### A.1. Безопасная обработка cost (C3)

```cpp
track_t dist = ...;
if (!std::isfinite(dist) || dist < 0)
    dist = maxPossibleCost;
dist = std::min(dist, maxPossibleCost);
costMatrix[i + j * colsTracks] = dist;
maxCost = std::max(maxCost, dist);
```

### A.2. Клэмпинг ROI перед гистограммой (C11)

```cpp
const cv::Rect frameRect(0, 0, currFrame.cols, currFrame.rows);
const cv::Rect roi = regions[j].m_brect & frameRect;
if (roi.width < 2 || roi.height < 2) continue;
std::vector<cv::UMat> regROI = { currFrame(roi) };
cv::calcHist(regROI, channels, cv::Mat(), regionEmbeddings[j].m_hist,
             histSize, ranges, false);
```

### A.3. Heat map (C4)

```cpp
const cv::Rect r = rect & cv::Rect(0, 0, m_heatMap.cols, m_heatMap.rows);
for (int y = r.y; y < r.y + r.height; ++y)
{
    float* row = m_heatMap.ptr<float>(y);
    for (int x = r.x; x < r.x + r.width; ++x)
        row[x] += w;
}
```

### A.4. `GetVelocity` (H2)

```cpp
const int rows = m_linearKalman.statePre.rows;
int vx = -1, vy = -1;
if (rows == 4)  { vx = 2; vy = 3; }
if (rows == 6)  { vx = 2; vy = 3; }   // [x,y,vx,vy,ax,ay]
if (rows == 8)  { vx = 4; vy = 5; }
if (rows == 10) { vx = 5; vy = 6; }   // [x,y,w,h,angle,vx,vy,...]
if (rows == 12) { vx = 4; vy = 5; }
```

### A.5. LAPJV (C2)

```cpp
for (col k = up; k < dimCols; ++k) { ... } // строки 183 и 215
```

---

## 9. Приложение B. Чек-лист перед релизом

- [ ] Ни одного `exit`/`assert(0)` в библиотечных путях (`grep -rn "exit(\|assert(0)" src`).
- [ ] Ни одного `nullptr`-разыменования в фабриках (`detector`, `m_detector`, `m_impl`).
- [ ] ASan/UBSan-прогон всех примеров без ошибок.
- [ ] Все комбинации `KalmanType × FilterGoal` дают либо результат, либо
      понятную ошибку.
- [ ] Python-модуль собирается и проходит smoke-тест.
- [ ] CI собирает Debug/Release, MSVC/GCC/Clang, Python и async-пример.
- [ ] README и `data/settings*.ini` соответствуют коду.
- [ ] Версии CMake/setup.py/`__version__` совпадают.
