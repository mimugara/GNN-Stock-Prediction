# Stock Prediction using Graph Neural Networks (GNN): Benchmarking of relational graphs of SP500 companies

---

**Bachelor Final Thesis (UPC-FIB), Spring 2024**
**Grade: 10/10**


---

### Abstract

In this thesis, a benchmark is developed for the evaluation of relational graphs
of SP500 companies for the stock market prediction problem. Multiple methods of constructing relational graphs are evaluated with industry sources, news,
structure, institutional investors, correlations of indicators, etc. To develop the
benchmark, a heterogeneous temporal Graph Neural Network has been reconstructed, which is able to beat the market consistently over time. 
Furthermore, a method of evaluating the graph by taking into account the correlation
graph of the following period has been proposed. In this way, the similarity
between graphs is measured and can facilitate the investigation of relational
graph construction methods in future research.


### Data preprocessing

First of all, the feature data and network generation must be generated. Since these files can be very heavy, they could not be included in the repository. 
In the *crawlers* folder they can be generated.

### Execution

After checking the *requirements.txt* file, the GNN can be executed through the *scheduler.py* file.

### Credits

- Author: Miquel Muñoz `miquelmunozz@gmail.com`

- Director: Sergi Abadal `sergiabadal@gmail.com`

- Codirector: Axel Wassington `axelwass@gmail.com`


## Citation

This thesis has reconstructed an existing architecture of a heterogeneous graph neural network. If you use this codebase, please cite the following paper.
The LICENSE of the project is GPL3, which means No matter how you modify or use code, you need to open source it.


```bibtex
@inproceedings{Xiang2022Temporal,
    author = {Xiang, Sheng and Cheng, Dawei and Shang, Chencheng and Zhang, Ying and Liang, Yuqi},
    title = {Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction},
    year = {2022},
    isbn = {9781450392365},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3511808.3557089},
    doi = {10.1145/3511808.3557089},
    booktitle = {Proceedings of the 31st ACM International Conference on Information & Knowledge Management},
    pages = {3584–3593},
    numpages = {10},
    location = {Atlanta, GA, USA},
    series = {CIKM '22}
}
