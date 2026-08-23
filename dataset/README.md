# Dataset

Put your labelled produce photos here in this exact layout (the class sub-folder
names become the model's output classes — add more folders to add classes):

```
dataset/
  train/       fresh/  ripe/  spoiled/
  validation/  fresh/  ripe/  spoiled/
  test/        fresh/  ripe/  spoiled/
```

See `docs/TRAINING.md` for image requirements, recommended sizes, the
train/validation/test split, augmentation, and evaluation. The `.gitkeep` files
just keep the empty folders in git; real images are gitignored by default.
