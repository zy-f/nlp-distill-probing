# Are Distilled Models Just Deep-Sea Octopi? Probing Linguistic Representations of Distillation-Finetuned Models
### Authors: Joy Yun (joyyun) and Christopher Polzak (zy-f)

CS 224N (Winter 2023) final project.
(Codebase includes significant contributions from both authors, but changes were committed from a shared cloud instance.)

[Paper](CS224N__Final_Report.pdf) | [Poster](CS224N__Final_Poster.pdf)

## Summary
In this project, we evaluate the effect of distilling BERT into DistilBERT during finetuning on the natural language inference (NLI) task, through word-level edge probing, generalization to benchmark NLI datasets, and generalization to challenge NLI datasets with intentionally modified function words. See the paper for more details.

Through our experiments, we find that:
- Performing distilled NLI finetuning consistently improves performance on in-distribution validation accuracy, and find that these improvements transfer to out-of-distribution generalization accuracy
- The teacher possesses a deeper linguistic understanding of function-word information that it then successfully transfers to the student
- Distillation can result in worsened understanding of word-level edge properties, but these declines are outweighed by the improvements distillation finetuning brings to other kinds of linguistic understanding (e.g. comprehension of function words)
- However, despite being less important, word-level edge understanding is beneficial for generalization performance

