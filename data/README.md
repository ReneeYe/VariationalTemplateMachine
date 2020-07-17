## Dataset
Two datasets (SPNLG and Wiki) can be downloaded from https://drive.google.com/drive/folders/1FsNlFh2aUbuBl45zEjgvAXDkp_e4hQmV?usp=sharing


## Statistics

<table>
  <tr>
    <th rowspan="2"></th>
    <th colspan="2">Train</th>
    <th colspan="2">Valid</th>
    <th>Test</th>
  </tr>
  <tr>
    <td>Paired</td>
    <td>Raw</td>
    <td>Paired</td>
    <td>Raw</td>
    <td>Paired</td>
  </tr>
  <tr>
    <td>SPNLG</td>
    <td>14k</td>
    <td>150k</td>
    <td>21k</td>
    <td>/</td>
    <td>21k</td>
  </tr>
  <tr>
    <td>Wiki</td>
    <td>84k</td>
    <td>842k</td>
    <td>73k</td>
    <td>43k</td>
    <td>73k</td>
  </tr>
</table>


## How we get the datasets?
- **SPNLG**
	- The dataset is from [sentence-planning-NLG dataset](https://nlds.soe.ucsc.edu/sentence-planning-NLG), a dataset describing the restaurant informations, containing 3 CSV files. 
	- We aggregate all the 3 CSV files, and leave `train:valid:test=8:1:1`, `paired:raw=1:10` for the train set.

- **Wiki**
	- The dataset is constructed from both [*Wiki-Bio* Dataset](https://github.com/DavidGrangier/wikipedia-biography-dataset) and [*Wikipedia Person and Animal* Dataset](https://drive.google.com/file/d/1TzcNdjZ0EsLh_rC1pBC7dU70jINcsVJd/view).
	- We used same valid and test set as *Wiki-Bio*.
	- For training set, we only randomly use 84k samples in *Wiki-Bio*-train for paired data. We use the remain sentences in *Wiki-Bio*-train and person descriptions from *Wikipedia Person and Animal* as raw data (totally up to 842k).

## Related links:
- Sentence planning NLG dataset: https://nlds.soe.ucsc.edu/sentence-planning-NLG
- Wikipedia biography dataset (Wiki-Bio): https://github.com/DavidGrangier/wikipedia-biography-dataset
- Wikipedia Person and Animal Dataset: https://drive.google.com/file/d/1TzcNdjZ0EsLh_rC1pBC7dU70jINcsVJd/view
