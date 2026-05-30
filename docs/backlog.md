# Uneemi ML - Backlog

Отложенные задачи и эксперименты. Не блокируют текущий milestone.

## После прохождения quality sanity

- [ ] INT8 dynamic quantization SigLIP 2 vision + re-bench
      `onnxruntime.quantization.quantize_dynamic`, ожидаемое ×2-4 ускорение,
      проверить деградацию на ImageNet zs (Δ ≤ 2 п.п.)

## После получения прод-сервера (4 vCPU)

- [ ] Performance benchmark на реальном VPS 4 vCPU
      Сравнить с dev-числами; зафиксировать прод-таргеты в `benchmark_results.md`

## Опционально / на будущее

- [ ] Full ImageNet val (50k) sanity benchmark
      Текущий stratified 5k даёт ±1 п.п. точности - для статьи или жёсткого
      паспорта нужен полный прогон (~6-8 ч на CPU)
- [ ] COCO 5k retrieval (image→text R@1)
      Дополнительная валидация retrieval-качества; время ~1 ч
- [ ] Flickr30k retrieval
      Альтернативный retrieval bench; время ~45 мин
- [ ] Sanity на полном XM3600 (все 36 языков)
      Сейчас валидируем только RU; для расширения географии нужен полный прогон
- [ ] CI: GitHub Actions с pytest на каждый PR
      После того, как появится коллаборация или прод
