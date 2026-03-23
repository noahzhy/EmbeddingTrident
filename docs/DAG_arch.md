```
                    ┌──────────────┐
Request  ─────────▶ │   Router     │
                    └──────┬───────┘
                           │
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
  Preprocess        Multi-Model         Postprocess
   (shared)            Layer              (shared)
```

