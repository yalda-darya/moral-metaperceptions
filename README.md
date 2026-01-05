# Asymmetric Moral Metaperceptions

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![NLP](https://img.shields.io/badge/Methods-NLP%20%7C%20Computational%20Social%20Science-brightgreen)
![Preregistered](https://img.shields.io/badge/Preregistered-Yes-important)
![OSF](https://img.shields.io/badge/OSF-Linked-lightgrey)

This repository contains all analysis code, figures, and supplementary materials for the paper:

> **Daryani, Y., Hejabi, P., and Dehghani, M.(2025). *Asymmetric Moral Metaperceptions: Consequences for Political Polarization and Paths to Correction*.**

The project investigates **moral metaperceptions** (i.e., people’s beliefs about how ideological outgroups evaluate their moral values) and shows how systematic inaccuracies in these beliefs contribute to political polarization, mistrust, and perceived threat while also demonstrating that these misperceptions are malleable and can be reduced through targeted feedback interventions.

---

## ⭐ Key Contributions

**Theoretical Contributions**

* Introduces **moral metaperceptions** as a distinct psychological construct: beliefs about how ideological outgroups evaluate one’s moral values.
* Demonstrates that moral metaperceptions exhibit **systematic asymmetries**, diverging from the uniform negativity bias documented in prior metaperception research.
* Shows that inaccurate moral metaperceptions uniquely predict **intergroup mistrust and perceived threat**, above and beyond general outgroup attitudes.
* Identifies moral metaperceptions as **malleable**, highlighting their potential as a target for polarization-reduction interventions.

**Methodological Contributions**

* Combines **large-scale NLP analysis** of millions of social media posts with **preregistered experimental designs**.
* Develops a discourse-based operationalization of **proto-metaperceptions** using opponent-directed moral language.
* Demonstrates how computational and experimental methods can be jointly leveraged to study moral cognition and polarization at scale.

---

## 📄 Paper Overview

Across **four studies**, this project combines large-scale computational social science with preregistered experiments:

* **Study 1 (Computational / NLP)**
  Analyzes millions of abortion-related tweets to examine how moral language differs when groups describe themselves versus their ideological opponents ("proto-metaperceptions").

* **Study 2 (Experiment – Abortion)**
  Tests the accuracy of moral metaperceptions between pro-life and pro-choice individuals, alongside warmth, competence, and social distance.

* **Study 3 (Experiment – Gun Control)**
  Replicates moral metaperceptual asymmetries in a new political domain and examines consequences for **trust** and **perceived threat**, with empathy as a moderator.

* **Study 4 (Intervention)**
  Tests whether corrective feedback about outgroup moral judgments reduces polarization by increasing trust and lowering threat.

In sum, the studies identify moral metaperceptions as a **distinct and malleable psychological mechanism** underlying political polarization.

---

## 📁 Repository Structure

```
.
├── study 1/
│ ├── code & analysis/ # NLP pipelines, classification models, regression analyses
│ ├── figures/ # Main and supplementary figures
│ └── supplementary/ # Annotation guides, robustness checks, additional analyses
│
├── study 2/
│ ├── code & analysis/ # Experimental analyses of moral metaperceptions
│ ├── figures/
│ └── supplementary/ # Extended results, additional contrasts, materials
│
├── study 3/
│ ├── code & analysis/ # Trust, threat, and moderation analyses
│ ├── figures/
│ └── supplementary/ # Exploratory analyses, material
│
├── study 4/
│ ├── code & analysis/ # Feedback intervention analyses
│ ├── figures/
│ └── supplementary/ # material
│
├── README.md # Project overview (this file)
├── LICENSE
└── .gitignore
```

Each study folder is self-contained and includes:

* Fully reproducible analysis scripts
* Model specifications and preprocessing steps
* Final figures used in the paper

---

## 🔬 Methods at a Glance

**Computational Methods (Study 1)**

* BERTweet fine-tuning for abortion stance classification
* RoBERTa-based moral foundation classifiers
* VADER sentiment analysis
* Binary and multinomial logistic regression (statsmodels)
* AI-based text analysis

**Experimental Methods (Studies 2–4)**

* Preregistered between-subjects experiments manipulating ingroup perception, outgroup perception, and metaperception conditions
* Scenario-based designs simulating real-world moral–political judgments across domains (abortion, gun control)
* Direct comparison of perceptions vs. metaperceptions to quantify metaperceptual accuracy and asymmetry
* Randomized feedback interventions providing corrective information about outgroup moral judgments
* Causal tests of downstream effects on trust and perceived threat, including moderation by empathy and perspective-taking

---

## 📊 Figures

All figures in this repository are **final, publication-ready versions** corresponding to those reported in manuscript. Each figure directory mirrors the structure of the Results sections in the paper.

---

## ♻️ Reproducibility

* All analyses were conducted in **Python** (computational studies) and **R** (experimental studies, as noted in each folder).
* Scripts are annotated to clarify preprocessing decisions, model choices, and statistical tests.

---

## 📎 Related Materials

* Full manuscript PDF: included with submission materials
* Preregistrations: linked in individual study folders (OSF)
* Annotation guidelines: available in Study 1 supplementary materials

---

## 👩‍🔬 Authors

**Yalda Daryani**
PhD Student, Social Psychology,
University of Southern California,
Morality & Language Lab

**Parsa Hejabi**
PhD Student, Computer Science,
University of Southern California,
Morality & Language Lab

**Morteza Dehghani**
Professor, Psychology and Computer Science,
University of Southern California,
Morality & Language Lab

---

## 📜 License

This repository is licensed under the terms specified in the `LICENSE` file.

---

If you use or build on this work, please cite the paper and link to this repository.
