# Classification Metrics and Class Imbalance

A model that calls every card payment honest is 99 percent accurate and catches no fraud at all. This repository is about measuring classifiers honestly when one class is rare, and about what to do once you can see the problem.

Notebooks 01 to 05 work with one dataset throughout: **card payments, about 1 in 100 of them fraudulent.** You start from why accuracy fails, build the confusion matrix, turn it into precision, recall and the F-scores, move the decision threshold, and finally try the standard ways of handling imbalance. The optional notebook 06 repeats the workflow on a harder problem, screening mammograms.

```
01_when_accuracy_lies           99% accurate, and useless
        |                       (the do-nothing baseline)
        v
02_confusion_matrix             four outcomes, two kinds of mistake
        |                       (which one costs more?)
        v
03_precision_recall_f_scores    how many caught, how many alarms real
        |                       (F1, F2, F0.5)
        v
04_thresholds_and_curves        the dial between the two mistakes
        |                       (precision-recall and ROC curves)
        v
05_handling_imbalance           class weights, resampling, thresholds
        |
        v
06_mammography_exercise         optional: all of it, on a new problem
                                (screening mammograms)
```

Both datasets ship in `data/`, so every notebook runs on its own and offline.

## Learning Objectives

By the end of this repository, you should be able to:

- Identify an imbalanced classification problem, and show with a majority-class baseline why accuracy misleads on it.
- Build and read a confusion matrix, and decide for a given problem whether a false positive or a false negative costs more.
- Compute precision, recall, F1 and F-beta from a confusion matrix, and choose among F1, F2 and F0.5 from the cost of each mistake.
- Move a classifier's decision threshold to trade precision against recall, choosing it on the training data rather than the test set.
- Read a precision-recall curve and an ROC curve, and explain why average precision is the more honest summary on imbalanced data.
- Apply class weights, random undersampling and threshold moving to an imbalanced problem, and compare them with a metric that fits its costs.

## Learning Path

**Scope** marks the core path every student is expected to complete. Optional lessons stay in the repository and are worth returning to, but the day does not depend on them. **Units** are indicative pacing, where one unit is 45 minutes.

| File / Folder | Description | Scope | Units |
| --- | --- | --- | --- |
| [**1 - When Accuracy Lies**](01_when_accuracy_lies.ipynb) | A do-nothing baseline at 99% accuracy against a logistic regression that catches four frauds in five, and where imbalance occurs. | Core | 1.25 |
| [**2 - The Confusion Matrix**](02_confusion_matrix.ipynb) | True and false positives and negatives, counted by hand and with scikit-learn, and which mistake costs more in three real problems. | Core | 1.5 |
| [**3 - Precision, Recall and F-Scores**](03_precision_recall_f_scores.ipynb) | Recall and precision from the four counts, why F1 is a harmonic mean, and F2 and F0.5 for when one mistake matters more. | Core | 1.75 |
| [**4 - Thresholds and Curves**](04_thresholds_and_curves.ipynb) | Moving the threshold, choosing one for a target recall on the training set, and the precision-recall and ROC curves. | Core | 1.75 |
| [**5 - Handling Imbalance**](05_handling_imbalance.ipynb) | Class weights, undersampling by hand with SMOTE as an idea, and a lower threshold, compared by precision, recall and F2, and a fourth strategy, models built for imbalance, named for later. | Core | 1.75 |
| [**6 - Mammography Exercise**](06_mammography_exercise.ipynb) | The whole workflow on a harder problem, step by step: a 98% accurate model that misses most cases, class weights, and a threshold chosen on the training set for a target recall. | Optional | 1.25 |

### Additional Folders and Files

| File / Folder | Description |
|---|---|
| [**Data**](data/) | `transactions.csv` for notebooks 01 to 05: 50,000 card payments, all 492 frauds of the original dataset and a random sample of the honest payments. `mammography.csv` for notebook 06: 11,183 image regions, 260 of them with a microcalcification. |
| [**Assets**](assets/) | The SMOTE figure used in notebook 05. |
| [**Solutions**](solutions/) | Reference solutions for every notebook. |
| [**pyproject.toml**](pyproject.toml) | Project configuration and dependencies. |
| [**uv.lock**](uv.lock) | Dependency lock file. |

## Setup

> [!NOTE]
> Throughout these steps, text in angle brackets like `<repo-name>` is a **placeholder**. Replace it, including the `< >` brackets, with your own value. For example, `cd <repo-name>` becomes `cd amle-metrics-class-imbalance`.

### 1. Create the Repository from the Template

Click **Use this template** on GitHub.

When creating the repository:

- Set yourself as the **Owner**
- Choose a repository name
- Disable **Include all branches**
- Click **Create repository**

> [!IMPORTANT]
> If you are working in pairs or groups, only **one person** should complete this step.

---

### 2. Add Collaborators (Pairs/Groups Only)

If working with teammates:

1. Open the repository on GitHub
2. Go to **Settings → Collaborators**
3. Add your teammates as collaborators
4. Share the repository link with your team

Teammates should accept the invitation before continuing.

---

### 3. Clone the Repository

Copy the SSH URL from the **Code** button on GitHub, then run:

```bash
git clone <copied-ssh-url>
```

The copied SSH URL will look like `git@github.com:<your-username>/<repo-name>.git`.

---

### 4. Move into the Project Folder and Install Dependencies

This installs all dependencies and creates a virtual environment in `.venv/`.

```bash
cd <repo-name>
uv sync
```

---

### 5. Open the Notebook

> [!NOTE]
> Make sure you open VS Code from the project root so it automatically detects the environment created by `uv sync`.

Launch VS Code in the project root folder:

```bash
code .
```

Then open `01_when_accuracy_lies.ipynb`, selecting the Python environment created by `uv sync` as the kernel. Work through the notebooks in order. Notebooks 01 to 05 are the core path, and 06 is optional practice on a new problem.

## References & Further Reading

- [**scikit-learn: Metrics and scoring**](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics): Every classification metric used here, with its definition.
- [**scikit-learn: Tuning the decision threshold**](https://scikit-learn.org/stable/modules/classification_threshold.html): Choosing a threshold properly, the idea behind notebook 04.
- [**Google Machine Learning Crash Course: Classification**](https://developers.google.com/machine-learning/crash-course/classification): Thresholds, the confusion matrix, precision, recall and ROC, with interactive exercises.
- [**Saito and Rehmsmeier, The Precision-Recall Plot Is More Informative than the ROC Plot**](https://doi.org/10.1371/journal.pone.0118432): The case for precision-recall curves on imbalanced data.
- [**imbalanced-learn**](https://imbalanced-learn.org/stable/user_guide.html): The standard library for resampling, including SMOTE, for when writing it by hand is no longer enough.
- [**OpenML: mammography**](https://www.openml.org/d/310): The dataset behind notebook 06, from a study of detecting microcalcifications in mammograms by Woods and colleagues.
- [**Credit Card Fraud Detection**](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): The original dataset of 284,807 payments, collected by the Machine Learning Group of ULB, Brussels.
