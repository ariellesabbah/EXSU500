# EXSU500_group_3
Group: Arielle Sabbah, Yarden Aronson, Sasha Socolow, Altara Wiryadi

## Dataset

This project uses the *BREAST-LESION-USG* dataset obtained from The Cancer Imaging Archive (TCIA)

Source:
Pawłowska, A., Ćwierz-Pieńkowska, A., Domalik, A., Jaguś, D., Kasprzak, P., Matkowski, R., Fura, Ł., Nowicki, A., & Zolek, N. (2024). A Curated Benchmark Dataset for Ultrasound Based Breast Lesion Analysis (Breast-Lesions-USG) (Version 1) [dataset]. The Cancer Imaging Archive. https://doi.org/10.7937/9WKK-Q141.

## Problem: 

Our project aims to develop and compare machine-learning models for predicting the malignancy of breast lesions from ultrasound images.

Why it matters: prompt and accurate diagnosis of breast lesions strongly influence follow-up and biopsy requirement. An AI model capable of predicting the malignancy of a lesion could potentially support more consistent and efficient image interpretation, and minimize unnecessary biopsies and associated complications.

## Task type: Binary Image Classification

Input:
Breast ultrasound image

Output:
Malignant or Benign

## Models compared

1. **Baselines:** majority class; logistic regression on lesion size and depth/width ratio
2. **Classical ML:** random forest and SVM on 17 handcrafted shape, intensity and texture features
3. **Deep learning:** ImageNet-pretrained ResNet-18, fine-tuned on the lesion region (3 seeds × 5 folds), with Grad-CAM

## Contributions (CRediT)

*To be completed*

## Use of generative AI

Claude (Anthropic) was used to draft starter code and that all code, results and citations were checked modified and are understood by the team.

## Licence

Code: MIT (see `LICENSE`). Data: CC-BY 4.0, owned by the dataset authors and not redistributed here.
