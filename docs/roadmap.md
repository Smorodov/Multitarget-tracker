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

> **Статус на 2026-11-16.** Исправленные пункты (C1–C14,
> H1–H19, H21–H23, M1, M2, M4, M5, M20–M23, M48; H20 закрыт как by-design без правок поведения)
> перенесены в `docs/roadmap-archive.md`. Полная история правок — в
> git-истории (батч стабильности a8fdcb5 от 2026-11-11, точечные фиксы
> 2026-11-15). Ниже — только открытая работа.

## 1. Легенда приоритетов

| Приоритет | Значение | Что делать |
|---|---|---|
| **P0** | Падение процесса, неопределённое поведение (UB), порча памяти, нерабочая заявленная функция, неразыменовываемый `nullptr` | Исправить в первую очередь, до любых релизов |
| **P1** | Неверный результат трекинга/детекции, зависание, исключение при валидной конфигурации, утечка/рост памяти | Исправить в ближайшем релизе |
| **P2** | Надёжность, производительность, переносимость сборки, неудобный/небезопасный API | Плановые работы |
| **P3** | Документация, мёртвый код, косметика, стиль | По возможности |

## 2. Сводка

### 2.1 P0 — критично

Открытых пунктов P0 нет — все критичные проблемы исправлены (см.
`docs/roadmap-archive.md`).

### 2.2 P1 — высокая важность

Открытых пунктов P1 нет — H18, H22 и H23 исправлены (см. `docs/roadmap-archive.md`).

### 2.3 P2/P3 — плановые

Ниже — сводные таблицы; подробные карточки для них не расписывались, но
направление исправления указано. Нумерация M сохранена из исходного
обзора (пропуски — пункты, закрытые правками из архива).

| ID | Приоритет | Кратко | Где |
|---|---|---|---|
| **M3** | P2 | `GetTracks` копирует все траектории (часть про `CalcEmbeddins` уже исправлена) | `BaseTracker.cpp:97-107`, `track.cpp:689-701` |
| **M6** | P2 | `DrawFilledRect`: `color[3]` при 4-канальном кадре и арифметика указателя при `rect.x >= cols` | `src/mtracking/defines.h:361-385` |
| **M7** | P2 | `nms/nms2/nms3`: деление на нулевую площадь объединения → NaN | `src/mtracking/nms.h:48-49,116-117,191-192` |
| **M8** | P2 | `WidthDist`/`HeightDist`: деление на 0 при нулевом размере | `src/Tracker/track.cpp:490-509` |
| **M9** | P2 | `TrackingObject::IsRobust` при пустой траектории: `size()-1` (underflow) и `0/0` → NaN | `src/Tracker/trajectory.h:260-284` |
| **M10** | P2 | `LeastSquarespoly2/2`: параметр `count` используется как конечный индекс; `sqrt` от отрицательного → NaN | `src/Tracker/trajectory.h:304-396` |
| **M11** | P2 | `BaseDetector::UpdateIgnoreMask`: отрицательный размер ROI при выходе за кадр | `src/Detector/BaseDetector.h:118-132` |
| **M12** | P2 | `BackgroundSubtract::Init` «успешно» возвращает `true` после fallback на VIBE; ошибки не видны вызывающему | `src/Detector/BackgroundSubtract.cpp:19-197` |
| **M13** | P2 | `std::stoi/stof` по значениям INI/KeyVal без `try/catch` → необработанное исключение | `MotionDetector.cpp:39`, `TrackerSettings.cpp`, `ONNXTensorRTDetector.cpp:50-63` |
| **M14** | P2 | `m_writeEachNFrame == 0` → деление на 0 при логировании | `example/FileLogger.h:113,149` |
| **M15** | P2 | `logs/` не создаётся в `VideoExample` → `rotating_file_sink` бросает исключение при старте (в `async_detector` каталог создаётся) | `example/VideoExample.cpp:55` |
| **M16** | P2 | `CarsCounting::RemoveLine` всегда возвращает `false`, даже удалив линию | `example/CarsCounting.cpp:391-401` |
| **M17** | P2 | `GeoParams::Pix2MapParams::Init`: при неудаче `m_toPix` остаётся неинициализированной | `example/CarsCounting.h:507-521` |
| **M18** | P2 | `VideoExample::AsyncProcess`: при `batch_size > 1` отображается/обрабатывается только первый кадр батча | `example/VideoExample.cpp:309-328` |
| **M19** | P2 | `AsyncDetector` измеряет не время обработки, а ожидание в очереди | `async_detector/AsyncDetector.cpp:77-90` |
| **M25** | P2 | `m_velocity` документирован как «pixels/sec», но это приращение за шаг фильтра (dt не равен реальному интервалу) | `trajectory.h:207`, `Kalman.cpp:1206` |
| **M27** | P3 | `CTrack` ctor №1 смещает первую точку траектории на `height/2`, ctor №2 — нет | `track.cpp:55` vs `110` |
| **M28** | P3 | `CreateLinear(RRect)` резервирует `m_initialRects` вместо `m_initialRRects` | `Kalman.cpp:230` |
| **M29** | P3 | `CreateLinear(Rect)`: `transitionMatrix(3,7) = dt/10` — несогласованно с остальными состояниями | `Kalman.cpp:107` |
| **M30** | P3 | Средний угол при инициализации rrect-модели — арифметическое среднее (неверно у ±90°) | `Kalman.cpp:1069-1076` |
| **M31** | P3 | `TrackID::ID2Module(0)` → деление на 0; `Str2ID` бросает `std::invalid_argument` | `defines.h:53-64` |
| **M32** | P3 | `TrackerSettings::EmbeddingParams` использует `assert` в конструкторе (в Release не проверяет) | `TrackerSettings.h:216-224` |
| **M34** | P3 | CVAT: жёстко `+00:00` при `localtime`; нет экранирования XML; `stop_frame = -1` при `framesCount = 0` | `example/cvatAnnotationsGenerator.h:183,199-208` |
| **M35** | P3 | `ResultsLog`: файл не открывается при имени ≤ 5 символов; CSV без ID трека | `example/FileLogger.h:29-38,109-125` |
| **M37** | P3 | README: `#include <mtracking/Ctracker.h>` (файла нет), устаревшие разделы/опции | `README.md:112` |
| **M38** | P3 | `data/settings.ini`: опечатка `inference_precison`; устаревшие `TrackMIL...`, `MatchBipart`, «speed in pixels» | `data/settings.ini:50,76-95,138` |
| **M39** | P3 | `data/settings_yolov8.ini` содержит абсолютные пути разработчика (`C:/work/...`) | `data/settings_yolov8.ini:29-31` |
| **M40** | P3 | CI: дефолтная сборка и Python 3.12; нет санитайзеров, интеграционных тестов, конфигураций TensorRT/UKF/CarsCounting | `.github/workflows/*.yml` |
| **M41** | P3 | `defines.h` тянет `opencv2/opencv.hpp` целиком, использует макросы `El_t`/`Mat_t`, не включает `<chrono>` напрямую | `src/mtracking/defines.h:1-29` |
| **M42** | P3 | `TrackingObject::IsRobust` меняет `mutable m_lastRobust` в константном методе — не потокобезопасно | `trajectory.h:214,260` |
| **M43** | P3 | `GetTrajectory` возвращает `vector<cv::Point>` (округление), теряя субпиксельную точность | `trajectory.h:290-298` |
| **M44** | P3 | `GetTracks` в `BYTETracker` строит `RotatedRect` из 3 точек — NaN при вырожденном прямоугольнике | `BYTETracker.cpp:34`, `defines.h:185-193` |
| **M45** | P3 | `AsyncDetector`/`VideoExample` дублируют код логгеров, отрисовки, CLI | `async_detector/*`, `example/*` |
| **M46** | P2 | Авто-определение входа по 4D-выходу (`DetectionOutput`) даёт `N×7` вместо размера сети; из-за дефолта `608` ветка обычно мертва | `OCVDNNDetector.cpp:282-327`, `OCVDNNDetector.h:68-69` |
| **M47** | P2 | `VerifyOutputAspectRatio` удаляет имена выходов из `m_params` во время обхода, а `m_outpuDims` не привязаны к именам → после удаления данные читаются как чужие тензоры | `YoloONNX.cpp:29-66,479-505` |

---

## 3. P1 — подробно

Открытых пунктов P1 нет — H18 и H22 исправлены 2026-11-15 (карточки и
подробности в `docs/roadmap-archive.md`).

---

## 4. P2/P3 — что делать по группам

### 4.1 Производительность

M1, M2, M4, M5 и часть M3 (`CalcEmbeddins`) исправлены 2026-11-15 —
карточки в `docs/roadmap-archive.md`.

* `GetTracks` копирует траектории (остаток M3). Для UI/логов достаточно
  `TrackingObject` без полной истории или с `shared_ptr` на неизменяемую
  траекторию.

### 4.2 Надёжность и численные методы

* Валидировать все входные `CRegion` (клэмпить `m_brect` к кадру,
  отбрасывать `width <= 0`/`height <= 0`) на входе `CTracker::Update` —
  закрывает оставшиеся NaN-ы и мусорные ROI из внешних источников.
* Валидировать `max_trace_len`, `max_lost_time`, `dt`, `min_static_time`,
  `minAreaRadius*` при парсинге настроек.
* `BackgroundSubtract::Init` должен возвращать `false`/код, если
  запрошенный алгоритм недоступен (M12).
* Обернуть парсинг INI/KeyVal в `try/catch` и давать понятные сообщения
  (M13).

### 4.3 API и архитектура

* Разделить `TrackerSettings` на `TrackerSettings` и `DetectorSettings`
  (или вложенные структуры), оставив совместимый парсер.
* Отделить публичные заголовки (`BaseTracker.h`, `TrackerSettings.h`,
  `trajectory.h`, типы) от внутренних (`track.h`, `Kalman.h`,
  `ShortPathCalculator.h`); обновить `PUBLIC_HEADER`/install.
* Ввести `DetectorFactory`-регистрацию вместо `switch` (упрощает
  добавление моделей и тестирование).
* Для парсеров DNN: выбирать выходы по именам, а не по индексам,
  проверять глубину/размеры тензоров (M46, M47).
* Привести `m_velocity` к задокументированным единицам (M25) или
  переименовать в `m_velocityPerStep`.

### 4.4 Сборка и упаковка

M20–M23 и CMake-пакет исправлены 2026-11-15 — карточки в
`docs/roadmap-archive.md`. Итог:

* флаги не перетирают пользовательские; `-march=native`/`/arch:AVX2` —
  под опцией `MT_NATIVE` (OFF по умолчанию, Release переносимый);
* единый тип библиотек через `MT_SHARED`; на MSVC при SHARED включены
  `CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS` и `MT_CORE_API` для реестра типов;
* реестр типов вынесен в `mtracking_core` (единый для трекера и
  детекторов), OpenCV линкуется `PUBLIC`;
* `Threads::Threads` и `OpenMP::OpenMP_CXX` — через target'ы, а не флаги;
* мёртвые include-пути удалены;
* CMake-пакет: `find_dependency(OpenCV/Threads/OpenMP)`, `LIB_DIR` из
  `LIB_INSTALL_DIR`, проверен consumer-проектом (static и shared).

### 4.5 Документация

* README: актуальный include (`BaseTracker.h`), актуальные опции сборки,
  пример Python, список поддерживаемых моделей и ограничений (M37).
* `data/settings*.ini`: убрать опечатки/устаревшие значения, не хранить
  абсолютные пути (M38, M39).
* Добавить `docs/` (этот файл и `architecture.md`) в README.
* Задокументировать единицы величин (dt, velocity, static time,
  minAreaRadius*, maxTraceLength).

---

## 5. План работ (roadmap по этапам)

### M0. Остаток P0 (1–2 недели)

Цель: убрать оставшиеся падения/UB/зависания; выпустить патч-релиз `1.1.1`.

- [x] H22 — реестр типов: границы, неизменяемость после старта,
      `bad_type` для неизвестных имён; заодно мёртвый `assert(0)`
      (`OCVDNNDetector.cpp:275`). Сделано 2026-11-15 (см.
      `docs/roadmap-archive.md`).
- [ ] Добавить ASan/UBSan-сборку в CI (хотя бы один workflow).

**Критерий готовности:** примеры `-e=1`, `-e=2`, `-e=4` и
`async_detector` запускаются под ASan на `data/atrium.avi` без ошибок;
недоступные/неизвестные бэкенды и типы не роняют процесс.

### M1. Тесты, CI, воспроизводимость

- [ ] Перенести локальные регрессионные наборы (`build/h*_test`,
      `build/c6_test`, сейчас вне git) в репозиторий под CTest:
      property-тест LAPJV, INI/`ParseTrackerSettings`, наборы h2–h22
      и c6.
- [ ] Юнит-тесты: LAPJV/Hungarian (прямоугольные и вырожденные матрицы),
      Kalman (модели/размерности), GEO (ENU ↔ WGS84, гомография),
      `nms`, `CRegion`, `TypeConverter`.
- [ ] Интеграционные: синтетическое видео с известными траекториями
      (2–3 сценария), проверка ID-switch, статики, подсчёта линий.
- [ ] CI-матрица: Linux GCC/Clang, Windows MSVC, macOS; Debug+Release;
      ASan/UBSan; `MTRACKER_PYTHON=ON`; `BUILD_ASYNC_DETECTOR=ON`;
      `BUILD_CARS_COUNTING=ON`; `USE_OCV_UKF=ON`; опционально TensorRT
      (self-hosted runner).
- [ ] Бенчмарк: 1080p, 30 fps, 50 объектов — бюджет кадра.

**Критерий готовности:** `ctest` запускается на CI на каждый PR;
регрессионные наборы живут в репозитории, а не в локальном `build/`.

### M2. Архитектурный рефакторинг (1–2 месяца)

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

### M3. Функциональное развитие (после стабилизации)

- [ ] Сегментационные маски для OpenCV DNN (сейчас только TensorRT).
- [ ] Полноценная поддержка `FilterRRect` (включая acceleration/UKF) или
      явный отказ.
- [ ] ReID: несколько моделей/типов, батч-инференс, кэширование.
- [x] Метрики MOT (MOTA/IDF1/ID-switch) как отдельный инструмент
      *(2026-11-16: `tools/mot_eval` — CLEAR MOT + ID-меры, семантика
      py-motmetrics, кросс-чек и юнит-тесты — см. карточку в архиве и
      `docs/mot_bench.md`)*.
- [x] Экспорт треков в MOTChallenge/CVAT (сейчас CVAT есть, MOT — нет)
      *(2026-11-16: `--mot_res=<file>` в `MultitargetTracker`, формат
      `frame,id,x,y,w,h,conf,-1,-1,-1`, 1-based кадры как в GT)*.
- [ ] GPU-декодирование/препроцессинг (опционально).

---

## 6. План тестирования (минимальный)

| Уровень | Что проверять | Инструмент | Статус |
|---|---|---|---|
| Unit | `lap()` vs эталон, `AssignmentProblemSolver`, `TKalmanFilter` (все модели), `CRegion` (вырожденные), `nms`, `TypeConverter`, `ParseTrackerSettings` | GoogleTest/Catch2 + CTest | есть INI-тесты `ParseTrackerSettings`; `TypeConverter` — локальный `build/h22_test` (не в git); LAPJV/Kalman/`nms`/`CRegion` — нет |
| Property | LAPJV: сумма cost ≤ Hungarian; перестановочность; прямоугольные матрицы | генератор случайных матриц | есть локально (`build/lapjv_test`), не в git/CI |
| Sanitizers | ASan/UBSan на примерах и unit-тестах | CI | нет (локальные регрессионные прогоны h2–h21 выполнялись) |
| Integration | Синтетическое видео: 5 объектов, пересечения, потеря, статика; проверка ID и counts | OpenCV + эталонные CSV | нет |
| Regression | `data/atrium.avi`, `data/TrackingBugs.mp4` — метрики и число треков | скрипт сравнения | частично: локальные наборы (h2–h22, c6) в gitignored `build/` |
| Fuzz | Парсеры выходов DNN (shape/тип), INI-парсер, `str2ID` | libFuzzer/AFL (опционально) | нет |
| Python | `import pymtracking`, `MTracker(settings, fps)`, `Update`, `GetTracks`, numpy-массивы | pytest | нет (wheel собирается, smoke — через `demo.py`) |

---

## 7. Приложение. Чек-лист перед релизом

- [ ] Ни одного `exit`/`assert(0)` в библиотечных путях
      (`grep -rn "exit(\|assert(0)" src`); в коде проекта таких вызовов
      нет — `OCVDNNDetector::Init` теперь возвращает `false` вместо
      `assert(0)` (H22); оставшиеся вхождения — в сторонних исходниках
      `src/Detector/tensorrt_onnx/common/` (вынос — задача M2).
- [ ] ASan/UBSan-прогон всех примеров без ошибок.
- [ ] Python-модуль собирается и проходит smoke-тест.
- [ ] CI собирает Debug/Release, MSVC/GCC/Clang, Python и async-пример.
- [ ] README и `data/settings*.ini` соответствуют коду.
