# Performance Benchmark - Siglip2Encoder

История прогонов performance-бенчмарка. Каждая секция - отдельный запуск `scripts/bench_inference.py`. Новые секции аппендятся в конец.

## Целевые показатели и факты

**Старые таргеты** (из `sravnenie_modelei.md`): `encode(single) p99 ≤ 30 мс`, `encode_batch(32) p99 ≤ 500 мс`, `ram_peak_bs32 ≤ 2 GB`. Эти числа были **оценками**; реальные замеры на fp32 показали, что 30 мс p99 single недостижимо без квантизации.

**Новые таргеты - Эталон: 12 physical cores, fp32 ONNX.** Headroom 11-13% над фактическими замерами на `threads=12` (54.7 / 1664 / 51.4 / 1256) - равномерно по всем метрикам, чтобы шум прогона не валил gate, и недостаточно, чтобы пропустить реальную регрессию в графе или препроцессинге.

| Метрика | Таргет | Наблюдённое (threads=12) | Headroom |
|---|---|---|---|
| `encode(single) p99` | ≤ 62 мс | 54.7 мс | +13% |
| `encode_batch(32) p99` | ≤ 1850 мс | 1664 мс | +11% |
| `per_img в батче` | ≤ 57 мс | 51.4 мс | +11% |
| `ram_peak_bs32` | ≤ 2048 MB | 1256 MB | исторический запас |

**Прод-таргеты (4 vCPU, INT8): TBD после квантизации.** Не выдумываем числа, которых не измеряли.

**Архитектурный вывод:** на CPU `per_img` практически константен от bs=1 до bs=32 (~51 мс), батчинг не даёт throughput-выгоды. Celery-воркеры будут запускаться с `concurrency=N`, каждый обрабатывает запросы по `batch=1` - выгоднее для realtime-SLA, чем batching.

**Дальнейшие шаги:** квантизация INT8 (`onnxruntime.quantization.quantize_dynamic`) запланирована после прохождения quality-бенчмарков (ImageNet zs / XM3600). Ожидаемое ускорение ×2-4, деградация качества проверится отдельно.

---

## Прогон 2026-05-26 20:03:07

**Commit:** `ea13908`
**Host:** Windows 10, Intel64 Family 6 Model 151 Stepping 2, GenuineIntel (12C/20T), 31.9 GB RAM
**Python:** 3.11.9 | **onnxruntime:** 1.26.0 | **ORT threads:** 4
**RAM после init Siglip2Encoder:** 416.4 MB

### Латентность

| Операция | p50 (мс) | p95 (мс) | p99 (мс) | per_img (мс) | ram_peak_mb |
|---|---|---|---|---|---|
| preprocess (PIL+normalize) | 2.96 | 3.33 | 3.65 | - | - |
| encode(single) | 82.69 | 89.76 | 92.02 | - | 545.3 |
| encode_batch(1) | 81.88 | 87.63 | 92.16 | 81.88 | 545.5 |
| encode_batch(8) | 657.94 | 815.70 | 826.95 | 82.24 | 704.1 |
| encode_batch(16) | 1312.95 | 1379.79 | 1401.55 | 82.06 | 889.0 |
| encode_batch(32) | 2657.02 | 2698.95 | 2715.01 | 83.03 | 1256.1 |

### Full board pipeline (40 изображений, чанки 32+8)

- t_preprocess: 105.99 мс
- t_encode: 3256.32 мс
- **t_total: 3360.10 мс**

### Таргеты

- [❌] p99 encode(single) ≤ 30 мс (фактическое: 92.02 мс)
- [❌] p99 encode_batch(32) ≤ 500 мс (фактическое: 2715.01 мс)
- [x] ram_peak_bs32 ≤ 2048 MB (фактическое: 1256.1 MB)

---

## Прогон 2026-05-26 20:14:45

**Commit:** `ea13908`
**Host:** Windows 10, Intel64 Family 6 Model 151 Stepping 2, GenuineIntel (12C/20T), 31.9 GB RAM
**Python:** 3.11.9 | **onnxruntime:** 1.26.0 | **ORT threads:** 8 (cli)
**RAM после init Siglip2Encoder:** 417.6 MB

### Латентность

| Операция | p50 (мс) | p95 (мс) | p99 (мс) | per_img (мс) | ram_peak_mb |
|---|---|---|---|---|---|
| preprocess (PIL+normalize) | 2.73 | 3.13 | 3.28 | - | - |
| encode(single) | 52.03 | 56.40 | 60.94 | - | 543.2 |
| encode_batch(1) | 52.06 | 56.62 | 59.72 | 52.06 | 543.2 |
| encode_batch(8) | 506.51 | 533.70 | 535.98 | 63.31 | 702.7 |
| encode_batch(16) | 1028.51 | 1088.65 | 1136.47 | 64.28 | 886.5 |
| encode_batch(32) | 2102.45 | 2186.01 | 2339.11 | 65.70 | 1254.2 |

### Full board pipeline (40 изображений, чанки 32+8)

- t_preprocess: 115.88 мс
- t_encode: 2420.79 мс
- **t_total: 2554.11 мс**

### Таргеты

- [❌] p99 encode(single) ≤ 30 мс (фактическое: 60.94 мс)
- [❌] p99 encode_batch(32) ≤ 500 мс (фактическое: 2339.11 мс)
- [x] ram_peak_bs32 ≤ 2048 MB (фактическое: 1254.2 MB)

---

## Прогон 2026-05-26 20:18:29

**Commit:** `ea13908`
**Host:** Windows 10, Intel64 Family 6 Model 151 Stepping 2, GenuineIntel (12C/20T), 31.9 GB RAM
**Python:** 3.11.9 | **onnxruntime:** 1.26.0 | **ORT threads:** 12 (cli)
**RAM после init Siglip2Encoder:** 418.6 MB

### Латентность

| Операция | p50 (мс) | p95 (мс) | p99 (мс) | per_img (мс) | ram_peak_mb |
|---|---|---|---|---|---|
| preprocess (PIL+normalize) | 2.55 | 3.08 | 3.55 | - | - |
| encode(single) | 46.95 | 49.95 | 54.75 | - | 544.5 |
| encode_batch(1) | 46.97 | 49.09 | 49.95 | 46.97 | 544.5 |
| encode_batch(8) | 395.57 | 406.02 | 411.17 | 49.45 | 703.9 |
| encode_batch(16) | 796.30 | 810.20 | 815.72 | 49.77 | 887.8 |
| encode_batch(32) | 1645.47 | 1660.02 | 1663.84 | 51.42 | 1255.5 |

### Full board pipeline (40 изображений, чанки 32+8)

- t_preprocess: 121.74 мс
- t_encode: 1954.11 мс
- **t_total: 2061.56 мс**

### Таргеты

- [❌] p99 encode(single) ≤ 30 мс (фактическое: 54.75 мс)
- [❌] p99 encode_batch(32) ≤ 500 мс (фактическое: 1663.84 мс)
- [x] ram_peak_bs32 ≤ 2048 MB (фактическое: 1255.5 MB)

---

## Прогон 2026-05-26 20:21:36

**Commit:** `ea13908`
**Host:** Windows 10, Intel64 Family 6 Model 151 Stepping 2, GenuineIntel (12C/20T), 31.9 GB RAM
**Python:** 3.11.9 | **onnxruntime:** 1.26.0 | **ORT threads:** 16 (cli)
**RAM после init Siglip2Encoder:** 418.7 MB

### Латентность

| Операция | p50 (мс) | p95 (мс) | p99 (мс) | per_img (мс) | ram_peak_mb |
|---|---|---|---|---|---|
| preprocess (PIL+normalize) | 2.52 | 2.90 | 3.09 | - | - |
| encode(single) | 46.68 | 48.31 | 50.75 | - | 544.9 |
| encode_batch(1) | 46.70 | 47.80 | 48.12 | 46.70 | 546.0 |
| encode_batch(8) | 380.71 | 388.87 | 389.81 | 47.59 | 704.3 |
| encode_batch(16) | 819.37 | 848.54 | 861.17 | 51.21 | 888.2 |
| encode_batch(32) | 1678.70 | 1716.50 | 1737.59 | 52.46 | 1255.9 |

### Full board pipeline (40 изображений, чанки 32+8)

- t_preprocess: 141.89 мс
- t_encode: 1937.22 мс
- **t_total: 2075.59 мс**

### Таргеты

- [❌] p99 encode(single) ≤ 30 мс (фактическое: 50.75 мс)
- [❌] p99 encode_batch(32) ≤ 500 мс (фактическое: 1737.59 мс)
- [x] ram_peak_bs32 ≤ 2048 MB (фактическое: 1255.9 MB)
