# datasets/

Put raw training datasets here (they are NOT shipped with this project).

* **UCI HAR** – download "UCI HAR Dataset" (https://archive.ics.uci.edu/dataset/240) and unzip into
  `datasets/UCI HAR Dataset/`. Train a source model with
  `python -m models.train --uci-har-dir "datasets/UCI HAR Dataset"`.
* **Other CSV datasets (WISDM, MotionSense, HHAR, custom)** – describe them in a JSON spec
  (see `datasets/example_spec.json`) and train with `python -m models.train --csv-spec datasets/my_spec.json`.
