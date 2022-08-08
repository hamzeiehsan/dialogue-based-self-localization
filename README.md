#An Entropy-based Model for Indoor Self-Localization through Dialogue
This is the implementation of an accepted short paper for the Conference on Spatial Information Theory (COSIT'22).
## Authors
* Kimia Amouzandeh: [email](kamoozandeh@student.unimelb.edu.au)
* Ehsan Hamzei: [email](ehsan.hamzei@unimelb.edu.au) 
* Martin Tomko: [email](tomkom@unimelb.edu.au), [website](tomko.org)

## Abstract
People can be localized at a particular location in an indoor environment using verbal descriptions referring to distinct visible objects (e.g., landmarks). When a user provides an incomplete initial location description their location may remain ambiguous. Here, we consider a dialogue initiated to update the initial description, which continues until the updated description can be related to a location in the environment. In each interaction, the wayfinder is incrementally asked about the visibility of a particular object to update the initial description. This paper presents an entropy-based model to minimize the number of interactions. We show how this entropy-based model leads to a significant reduction of interactions (i.e., reduction of conversation length, measured by the number of additional referents) compared to baseline models. Moreover, the effect of the initial description, i.e., the first set of visible objects with different combinations, is investigated.

## Workflow
### Calculate conversation length for the method and baseline

```python
python conversation.py
```

### Generating the figures 
```python
python result_analysis.py
python AVG_Comparison.py
```