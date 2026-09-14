# Dataset registry

No datasets have been evaluated under this benchmark yet. Copy the template
below for each dataset and evaluation case. Replace placeholders with verified
metadata before running experiments; unknown information must be marked unknown.

## Selected benchmark datasets

The following datasets are selected for future benchmark runs. Categories are
provided by the benchmark authors and have not yet been independently verified.
A dataset can appear in multiple categories; repeated OpenML IDs refer to the
same dataset, not separate experiments. There are 22 unique datasets.

Before evaluation, complete the entry template below for each dataset/case,
including version, size, license, and evidence for expected findings. These
selection categories do not establish ground-truth issues or experimental results.

### Binary classification

| OpenML ID | Source |
| --- | --- |
| 1464 | [OpenML 1464](https://www.openml.org/search?type=data&status=active&id=1464) |
| 42900 | [OpenML 42900](https://www.openml.org/search?type=data&status=active&id=42900) |
| 45578 | [OpenML 45578](https://www.openml.org/search?type=data&status=active&id=45578) |
| 293 | [OpenML 293](https://www.openml.org/search?type=data&status=active&id=293) |
| 1480 | [OpenML 1480](https://www.openml.org/search?type=data&status=active&id=1480) |
| 44269 | [OpenML 44269](https://www.openml.org/search?type=data&status=active&id=44269) |
| 1018 | [OpenML 1018](https://www.openml.org/search?type=data&status=active&id=1018) |

### Multiclass classification

| OpenML ID | Source |
| --- | --- |
| 43750 | [OpenML 43750](https://www.openml.org/search?type=data&status=active&id=43750) |
| 1106 | [OpenML 1106](https://www.openml.org/search?type=data&status=active&id=1106) |
| 300 | [OpenML 300](https://www.openml.org/search?type=data&status=active&id=300) |
| 46600 | [OpenML 46600](https://www.openml.org/search?type=data&status=active&id=46600) |
| 40982 | [OpenML 40982](https://www.openml.org/search?type=data&status=active&id=40982) |

### Regression

| OpenML ID | Source |
| --- | --- |
| 43071 | [OpenML 43071](https://www.openml.org/search?type=data&status=active&id=43071) |
| 41704 | [OpenML 41704](https://www.openml.org/search?type=data&status=active&id=41704) |
| 44031 | [OpenML 44031](https://www.openml.org/search?type=data&status=active&id=44031) |
| 206 | [OpenML 206](https://www.openml.org/search?type=data&status=active&id=206) |

### Missing values

| OpenML ID | Source |
| --- | --- |
| 41704 | [OpenML 41704](https://www.openml.org/search?type=data&status=active&id=41704) |
| 44084 | [OpenML 44084](https://www.openml.org/search?type=data&status=active&id=44084) |
| 1018 | [OpenML 1018](https://www.openml.org/search?type=data&status=active&id=1018) |

### High-dimensional

| OpenML ID | Source |
| --- | --- |
| 1106 | [OpenML 1106](https://www.openml.org/search?type=data&status=active&id=1106) |
| 300 | [OpenML 300](https://www.openml.org/search?type=data&status=active&id=300) |
| 1038 | [OpenML 1038](https://www.openml.org/search?type=data&status=active&id=1038) |
| 40910 | [OpenML 40910](https://www.openml.org/search?type=data&status=active&id=40910) |

### Small-n

| OpenML ID | Source |
| --- | --- |
| 42900 | [OpenML 42900](https://www.openml.org/search?type=data&status=active&id=42900) |
| 43750 | [OpenML 43750](https://www.openml.org/search?type=data&status=active&id=43750) |
| 1106 | [OpenML 1106](https://www.openml.org/search?type=data&status=active&id=1106) |
| 206 | [OpenML 206](https://www.openml.org/search?type=data&status=active&id=206) |
| 691 | [OpenML 691](https://www.openml.org/search?type=data&status=active&id=691) |

### Outlier

| OpenML ID | Source |
| --- | --- |
| 42793 | [OpenML 42793](https://www.openml.org/search?type=data&status=active&id=42793) |
| 45081 | [OpenML 45081](https://www.openml.org/search?type=data&status=active&id=45081) |
| 40910 | [OpenML 40910](https://www.openml.org/search?type=data&status=active&id=40910) |

## Dataset entry template

| Field | Value |
| --- | --- |
| Dataset ID | <stable identifier> |
| Dataset name | <name> |
| Source | <canonical URL, citation, and download instructions> |
| Version and retrieval date | <release/version and UTC date> |
| Integrity | <file names and SHA-256 checksums> |
| Task type | <classification, regression, clustering, etc.> |
| Target and inputs | <target, features, and relevant artifacts> |
| Size | <rows, columns, and bytes before and after preparation> |
| License | <license identifier/name, source link, and access/redistribution terms> |
| Case ID | <stable identifier for this variant or evaluation scenario> |
| Preparation | <exact commands, dependencies, transformations, and seeds> |
| Split | <train/validation/test rule and saved indices or hashes, if applicable> |
| Evaluation input | <files and context exposed to MLCompass> |
| Known issues | <issue IDs below, or verified clean scope> |
| Expected findings | <findings required for each known issue> |

### Ground-truth issue template

| Issue ID | Known issue and affected scope | Evidence/source | Expected finding and matching rule |
| --- | --- | --- | --- |
| <unique ID> | <problem, affected columns/rows/artifacts> | <citation or reproducible verification> | <minimum specific finding needed for credit> |

Document whether each issue is naturally present or deliberately introduced.
For introduced issues, preserve the original checksum, transformation commands,
seed, and resulting checksum. Treat each variant as a separate case.

### Scope and uncertainty

- Verified absent issues / negative controls: <scope and supporting checks>.
- Unverified or disputed issues: <uncertainties excluded from ground truth>.
- Limitations: <what cannot be assessed from the supplied input>.

Freeze the issue list and matching rules before inspecting model outputs. A
public dataset's reputation alone does not establish that an issue exists.
Keep expected findings and ground-truth annotations out of model-visible input.
