# Tổng hợp Biomedical Knowledge Graphs: OREGANO, CKG, MonarchKG và PrimeKG

> **Mục đích:** Tài liệu tham khảo nhanh về bốn biomedical knowledge
> graphs (BKGs), gồm mục tiêu, phạm vi dữ liệu, cấu trúc, publication và
> khả năng ứng dụng trong Biomedical NLP / KGQA / graph reasoning.
>
> **Lưu ý về số liệu:** Quy mô nodes/edges có thể thay đổi theo phiên
> bản, bản phát hành và cách đếm (nodes, edges, associations hoặc
> triples). Các con số dưới đây là số liệu tham khảo từ
> publication/phiên bản được nêu, không nên xem là thống kê live của
> dataset.

## 1. Biomedical Knowledge Graph là gì?

**Biomedical Knowledge Graph (BKG)** là đồ thị tri thức biểu diễn các
thực thể y sinh và quan hệ giữa chúng.

Một KG thường được biểu diễn dưới dạng triples:

```text
(head entity, relation, tail entity)
```

Ví dụ minh họa:

```text
(Drug A, treats, Disease B)
(Protein X, associated_with, Disease B)
(Gene Y, encodes, Protein X)
```

-   **Nodes / entities:** disease, drug, gene, protein, phenotype,
    pathway, anatomy, publication, v.v.
-   **Edges / relations:** treats, associated_with, interacts_with,
    causes, participates_in, has_phenotype, v.v.
-   **Attributes / metadata:** identifier, source database, evidence,
    species, description, provenance, confidence score, v.v.

BKGs thường được tạo bằng cách tích hợp nhiều nguồn dữ liệu chuyên
ngành. Quá trình này có thể bao gồm data extraction, entity
normalization, entity resolution, relation mapping, ontology alignment
và graph construction.

## 2. Bảng so sánh nhanh

| Knowledge Graph | Publication chính | Trọng tâm | Quy mô tham khảo | Điểm nổi bật |
| --- | --- | --- | --- | --- |
| **OREGANO** | 2023 | Drug repurposing | 88,937 nodes; 824,231 edges | Tích hợp thuốc, target và natural compounds |
| **CKG** | 2022 | Clinical proteomics | Khoảng 16–20 triệu nodes và 220 triệu relationships, theo thống kê được trích dẫn trong tài liệu thứ cấp | KG kết hợp dữ liệu proteomics, clinical và biomedical |
| **MonarchKG** | Monarch Initiative; KG v3 ra mắt 2023; bài báo nền tảng 2024 | Gene–disease–phenotype, rare disease và cross-species | Số liệu phụ thuộc phiên bản; bài báo/nguồn thứ cấp có các thống kê khác nhau | Chuẩn hóa theo Biolink Model, hỗ trợ phenotype-driven discovery |
| **PrimeKG** | 2023 (preprint 2022) | Precision medicine, disease-centric reasoning | 129,375 nodes; 4,050,249 edges | Multimodal, 10 nhóm node chính, nhiều loại quan hệ |

**Cách đọc bảng:** Không nên so sánh trực tiếp quy mô của các KG nếu
chưa kiểm tra phiên bản, định nghĩa edge/association, mức độ lọc và phạm
vi dữ liệu. Một graph lớn hơn không nhất thiết phù hợp hơn cho một bài
toán cụ thể.

------------------------------------------------------------------------

## 3. OREGANO

### 3.1. Thông tin publication

-   **Tên bài báo:** *The OREGANO knowledge graph for computational drug
    repurposing*
-   **Năm xuất bản:** 2023
-   **Journal:** Scientific Data
-   **Volume / article:** Volume 10, Article 871
-   **DOI:**
    [10.1038/s41597-023-02757-0](https://doi.org/10.1038/s41597-023-02757-0)
-   **Trọng tâm:** Computational drug repurposing, bao gồm cả natural
    compounds.

### 3.2. Mục tiêu và bài toán

Drug repurposing (drug repositioning) là quá trình tìm kiếm công dụng
điều trị mới cho một loại thuốc hoặc hợp chất đã biết.

OREGANO kết nối thông tin về compounds/drugs với các thực thể y sinh như
protein, gene, disease, pathway, phenotype và side effect. Các liên kết
trong graph có thể được khai thác để tìm các quan hệ tiềm năng, ví dụ
drug--target hoặc drug--disease.

Ví dụ câu hỏi nghiên cứu:

-   Một compound tác động lên những protein/target nào?
-   Những target đó liên quan đến disease nào?
-   Có thể tìm thấy drug--disease association tiềm năng nào chưa được
    ghi nhận không?

### 3.3. Quy mô và cấu trúc

Số liệu tham khảo được báo cáo:

-   **Nodes:** 88,937
-   **Edges:** 824,231
-   **Node types:** khoảng 12
-   **Edge categories:** khoảng 19

Các nhóm thực thể tiêu biểu:

-   Drug / compound
-   Protein / target
-   Gene
-   Disease
-   Pathway
-   Phenotype
-   Side effect

Các nguồn dữ liệu được sử dụng trong hệ sinh thái OREGANO bao gồm những
nguồn như DrugBank, UniProt, Human Phenotype Ontology, PharmGKB, NPASS,
SIDER và Reactome. Cần kiểm tra publication và bản phát hành cụ thể để
xác nhận chính xác nguồn nào đóng góp vào từng loại node/edge.

### 3.4. Điểm nổi bật

1.  Tập trung vào drug repurposing.
2.  Có tích hợp natural products/compounds.
3.  Kết nối drug/compound với target và disease-related knowledge.
4.  Có thể dùng làm nền tảng cho link prediction và drug--target
    analysis.

### 3.5. Use cases

-   Drug repurposing
-   Drug--target prediction
-   Link prediction
-   Compound-centric biomedical discovery

### 3.6. Tài liệu

-   [Publication / DOI](https://doi.org/10.1038/s41597-023-02757-0)

------------------------------------------------------------------------

## 4. CKG — Clinical Knowledge Graph

### 4.1. Thông tin publication

-   **Tên bài báo:** *A knowledge graph to interpret clinical proteomics
    data*
-   **Năm xuất bản:** 2022
-   **Journal:** Nature Biotechnology
-   **Volume / pages:** Volume 40, pages 692--702
-   **DOI:**
    [10.1038/s41587-021-01145-6](https://doi.org/10.1038/s41587-021-01145-6)
-   **Trọng tâm:** Tích hợp và diễn giải dữ liệu clinical proteomics.

### 4.2. Mục tiêu và bài toán

Trong nghiên cứu y sinh, dữ liệu proteomics, clinical variables,
protein, gene, disease, drug và scientific literature thường nằm ở các
nguồn riêng biệt.

CKG kết nối những thông tin này trong một knowledge graph và cung cấp
một platform hỗ trợ phân tích dữ liệu proteomics, diễn giải kết quả thí
nghiệm và tạo giả thuyết nghiên cứu.

Ví dụ:

Một thí nghiệm phát hiện protein X có biểu hiện khác biệt ở một nhóm
bệnh nhân. Nhà nghiên cứu có thể cần tìm hiểu:

1.  Protein X có liên quan đến disease nào?
2.  Protein X tương tác với protein nào?
3.  Protein X tham gia pathway nào?
4.  Những publication nào đã đề cập đến các mối quan hệ đó?
5.  Có thể xem xét protein này như một biomarker tiềm năng không?

### 4.3. Quy mô và cấu trúc

Một số thống kê thứ cấp được trích dẫn cho CKG:

-   **Nodes:** khoảng 16--20 triệu
-   **Relationships:** khoảng 220 triệu
-   **Node types:** khoảng 19
-   **Edge types:** khoảng 57
-   **Nguồn tích hợp:** khoảng 26 biomedical databases và 9 ontologies

Các con số này cần được đối chiếu với phiên bản CKG cụ thể trước khi
dùng trong báo cáo hoặc publication.

Những nhóm entity tiêu biểu:

-   Protein
-   Gene
-   Drug
-   Disease
-   Publication
-   Clinical variables
-   Experimental data

### 4.4. Điểm nổi bật

1.  Không chỉ là một tập triples tĩnh; CKG gắn với một platform phân
    tích dữ liệu.
2.  Tập trung vào clinical proteomics.
3.  Kết nối dữ liệu thí nghiệm với kiến thức biomedical có sẵn.
4.  Hỗ trợ phân tích, diễn giải và tạo giả thuyết.

### 4.5. Use cases

-   Clinical proteomics
-   Biomarker discovery
-   Biomedical data integration
-   Clinical research support
-   Hypothesis generation

### 4.6. Tài liệu

-   [Publication / DOI](https://doi.org/10.1038/s41587-021-01145-6)
-   [CKG documentation / project
    resources](https://github.com/MannLabs/CKG)

------------------------------------------------------------------------

## 5. MonarchKG — Monarch Knowledge Graph

### 5.1. Thông tin publication và phiên bản

-   **Tên bài báo liên quan:** *The Monarch Initiative in 2024: an
    analytic platform integrating phenotypes, genes and diseases across
    species*
-   **Năm xuất bản:** 2024 (Nucleic Acids Research Database issue)
-   **DOI:**
    [10.1093/nar/gkad1082](https://doi.org/10.1093/nar/gkad1082)
-   **Thông tin phiên bản:** Monarch Knowledge Graph v3 được công bố vào
    năm 2023.
-   **Trọng tâm:** Kết nối phenotype, gene, disease và dữ liệu sinh học
    giữa nhiều loài.

**Lưu ý:** MonarchKG là một knowledge graph được phát triển liên tục
trong hệ sinh thái Monarch Initiative, không phải dataset chỉ tồn tại từ
năm 2024. Khi trích dẫn, nên phân biệt năm của publication, ngày phát
hành phiên bản và ngày snapshot dữ liệu.

### 5.2. Mục tiêu và bài toán

MonarchKG tập trung vào việc kết nối:

-   Gene
-   Disease
-   Phenotype
-   Genetic variants
-   Anatomy
-   Biological processes / pathways
-   Chemical entities

Một ứng dụng quan trọng là phenotype-driven disease discovery, đặc biệt
trong nghiên cứu rare diseases.

Ví dụ:

Một bệnh nhân có các phenotype như developmental delay, intellectual
disability và một số bất thường của hệ thần kinh. Hệ thống có thể khai
thác các liên kết phenotype--disease--gene để tìm những disease hoặc
gene ứng viên.

### 5.3. Data model và quy trình xây dựng

Một điểm nổi bật của Monarch là sử dụng **Biolink Model** để chuẩn hóa
cách biểu diễn biomedical entities và relationships.

Các thành phần/công cụ trong hệ sinh thái xây dựng graph có thể bao gồm:

-   **Biolink Model:** mô hình dữ liệu chuẩn hóa entities và relations.
-   **Koza:** framework hỗ trợ chuyển đổi dữ liệu nguồn thành graph
    data.
-   **KGX:** công cụ và định dạng hỗ trợ thao tác, trao đổi knowledge
    graph.
-   **Cat-Merge:** công cụ trong quy trình hợp nhất dữ liệu/graph.

Quy trình khái quát:

```text
Biomedical data sources
        |
        v
Source-specific transformation
        |
        v
Normalization / mapping to Biolink Model
        |
        v
Merge and resolve entities
        |
        v
Monarch Knowledge Graph
        |
        v
Query, phenotype matching, graph analysis
```

### 5.4. Quy mô

Không nên gán một con số nodes/edges duy nhất cho MonarchKG mà không ghi
rõ snapshot/version.

Các thống kê được báo cáo ở các thời điểm khác nhau có thể khác nhau
đáng kể. Một số nguồn mô tả phiên bản 2024 ở mức khoảng một triệu nodes
và khoảng 11 triệu associations; các dashboard/bản cập nhật sau đó có
thể lớn hơn.

**Khuyến nghị:** Nếu cần số liệu chính xác cho nghiên cứu, hãy ghi lại
ngày tải dữ liệu, release/version, số nodes, số edges và cách đếm từ
chính snapshot đã sử dụng.

### 5.5. Điểm nổi bật

1.  Hỗ trợ cross-species knowledge integration.
2.  Chuẩn hóa graph theo Biolink Model.
3.  Kết nối gene--disease--phenotype.
4.  Phù hợp với phenotype matching và rare-disease research.
5.  Có hệ sinh thái công cụ phục vụ xây dựng và truy vấn KG.

### 5.6. Use cases

-   Rare disease diagnosis
-   Phenotype matching
-   Gene--disease association
-   Cross-species reasoning
-   Biomedical semantic search
-   Graph-based discovery

### 5.7. Tài liệu

-   [Monarch Initiative 2024
    paper](https://doi.org/10.1093/nar/gkad1082)
-   [Monarch Initiative](https://monarchinitiative.org/)
-   [Monarch Knowledge Graph
    documentation](https://monarch-initiative.github.io/monarch-ingest/)

------------------------------------------------------------------------

## 6. PrimeKG — Precision Medicine Knowledge Graph

### 6.1. Thông tin publication

-   **Tên bài báo:** *Building a knowledge graph to enable precision
    medicine*
-   **Authors:** Payal Chandak, Kexin Huang, Marinka Zitnik
-   **Năm xuất bản:** 2023
-   **Journal:** Scientific Data
-   **Volume / article:** Volume 10, Article 67
-   **DOI:**
    [10.1038/s41597-023-01960-3](https://doi.org/10.1038/s41597-023-01960-3)
-   **Preprint:** 2022
-   **Trọng tâm:** Multimodal, disease-centric biomedical knowledge
    graph phục vụ precision medicine.

### 6.2. Mục tiêu và bài toán

Một disease có thể liên quan đồng thời đến gene, protein, biological
processes, molecular pathways, phenotypes, drugs, side effects, anatomy
và environmental exposures.

PrimeKG kết nối nhiều khía cạnh đó trong một graph thống nhất để hỗ trợ
nghiên cứu precision medicine.

Ví dụ, với một disease, nhà nghiên cứu có thể tìm:

1.  Những protein nào liên quan đến disease?
2.  Những protein đó tham gia pathway nào?
3.  Disease có những phenotype nào?
4.  Những drug nào có quan hệ điều trị hoặc chống chỉ định?
5.  Những side effects nào liên quan đến drug?

### 6.3. Quy mô và cấu trúc

Số liệu publication:

-   **Nodes:** 129,375
-   **Edges:** 4,050,249
-   **Biomedical resources:** 20
-   **Node types:** 10 nhóm chính
-   **Relation types:** khoảng 30 loại

Mười nhóm node chính:

| Node type | Ý nghĩa |
| --- | --- |
| Disease | Bệnh |
| Drug | Thuốc |
| Protein | Protein |
| Biological Process | Quá trình sinh học |
| Molecular Function | Chức năng phân tử |
| Cellular Component | Thành phần tế bào |
| Pathway | Con đường sinh học |
| Phenotype | Kiểu hình |
| Anatomy | Cấu trúc giải phẫu |
| Exposure | Yếu tố phơi nhiễm |

Các quan hệ tiêu biểu gồm disease--drug, disease--protein,
protein--pathway, disease--phenotype và drug--side effect. Cần kiểm tra
schema của bản dataset để lấy đúng tên relation và chiều cạnh.

### 6.4. Điểm nổi bật

#### 6.4.1. Disease-centric

Disease là một thực thể trung tâm, được kết nối với nhiều loại
biomedical entities.

#### 6.4.2. Quan hệ drug–disease phong phú

Các quan hệ có thể biểu diễn thông tin như:

-   Indication: thuốc được chỉ định cho disease.
-   Contraindication: thuốc chống chỉ định với disease.
-   Off-label use: sử dụng thuốc ngoài chỉ định được phê duyệt.

#### 6.4.3. Multimodal

PrimeKG không chỉ chứa graph structure mà còn cung cấp các mô tả dạng
văn bản liên quan đến clinical guidelines cho disease và drug nodes. Đây
là một đặc điểm có ích khi kết hợp graph với language models.

### 6.5. Use cases

-   Precision medicine
-   Drug repurposing
-   Biomedical KGQA
-   Multi-hop graph reasoning
-   Disease understanding
-   Link prediction
-   Biomedical AI

### 6.6. Tài liệu

-   [Publication / DOI](https://doi.org/10.1038/s41597-023-01960-3)
-   [PrimeKG project / dataset](https://github.com/mims-harvard/PrimeKG)

------------------------------------------------------------------------

## 7. So sánh theo mục tiêu nghiên cứu

| Tiêu chí | OREGANO | CKG | MonarchKG | PrimeKG |
| --- | --- | --- | --- | --- |
| Drug repurposing | Trọng tâm | Có thể khai thác | Có thể khai thác | Có |
| Clinical proteomics | Không phải trọng tâm | Trọng tâm | Không phải trọng tâm | Không phải trọng tâm |
| Rare disease | Không phải trọng tâm | Có thể khai thác | Trọng tâm quan trọng | Có thể khai thác |
| Gene–disease | Có | Có | Trọng tâm | Có |
| Phenotype reasoning | Có | Có thể khai thác | Trọng tâm | Có |
| Cross-species | Không phải trọng tâm | Không phải trọng tâm | Có | Không phải trọng tâm |
| Multi-hop KGQA | Có thể dùng | Có thể dùng | Có thể dùng | Phù hợp |
| Link prediction | Có | Có thể áp dụng | Có thể áp dụng | Có thể áp dụng |
| Clinical data integration | Hạn chế | Trọng tâm | Có dữ liệu liên quan | Có thông tin clinical guideline |

### 7.1. Cách phân biệt ngắn gọn

-   **OREGANO -- Drug-centric:** Tập trung vào compound/drug, target và
    drug repurposing.
-   **CKG -- Proteomics/clinical-centric:** Tập trung vào dữ liệu
    proteomics và diễn giải dữ liệu nghiên cứu lâm sàng.
-   **MonarchKG -- Gene/phenotype/disease-centric:** Tập trung vào
    phenotype, gene, disease, rare disease và cross-species.
-   **PrimeKG -- Precision-medicine/disease-centric:** Kết nối disease
    với nhiều khía cạnh sinh học và điều trị.

------------------------------------------------------------------------

## 8. Liên hệ với Biomedical KGQA và Multi-hop Reasoning

Việc lựa chọn KG ảnh hưởng trực tiếp đến loại câu hỏi và reasoning path
mà hệ thống KGQA có thể khai thác.

Ví dụ các câu hỏi nghiên cứu:

| KG | Ví dụ câu hỏi |
| --- | --- |
| OREGANO | Which proteins are targeted by compounds associated with a particular disease? |
| CKG | Which biological pathways are associated with a protein observed in a clinical proteomics study? |
| MonarchKG | Which genes are associated with a disease characterized by a given set of phenotypes? |
| PrimeKG | Which drugs are associated with a disease whose related proteins participate in a particular pathway? |

### 8.1. Ví dụ multi-hop reasoning với PrimeKG

Câu hỏi minh họa:

> Which drugs are associated with a disease that is linked to protein X?

Reasoning path khái quát:

```text
Protein X
   |
   | associated_with
   v
Disease
   |
   | treated_by / indication
   v
Drug
```

Các bước:

1.  Tìm disease có quan hệ với Protein X.
2.  Từ disease đó, truy vấn các drug có quan hệ điều trị.
3.  Tổng hợp các drug thỏa mãn reasoning path.

Đây chỉ là ví dụ thiết kế câu hỏi; cần kiểm tra relation thực tế và
chiều edge trong schema của PrimeKG trước khi chạy query.

### 8.2. Những yếu tố cần kiểm tra khi chọn KG cho KGQA

1.  **Entity coverage:** KG có chứa các loại entity cần thiết cho câu
    hỏi không?
2.  **Relation diversity:** Có đủ loại quan hệ để tạo multi-hop paths
    không?
3.  **Graph connectivity:** Các entity có được kết nối đủ tốt không?
4.  **Entity normalization:** Entity có identifier chuẩn và có xử lý
    trùng lặp không?
5.  **Provenance:** Có biết dữ liệu đến từ nguồn nào và bằng chứng nào
    không?
6.  **Versioning:** Có thể xác định chính xác snapshot/version đã dùng
    không?
7.  **Textual descriptions:** Có mô tả văn bản để kết hợp với LLM không?
8.  **License and accessibility:** Có thể tải, sử dụng và công bố kết
    quả theo điều kiện license không?

------------------------------------------------------------------------

## 9. Khuyến nghị lựa chọn dataset

| Mục tiêu nghiên cứu | Dataset nên xem xét trước |
| --- | --- |
| Drug repurposing / drug–target prediction | OREGANO |
| Clinical proteomics / biomarker discovery | CKG |
| Rare disease / phenotype-driven diagnosis | MonarchKG |
| Precision medicine / multi-hop biomedical KGQA | PrimeKG |
| Cross-species biological reasoning | MonarchKG |
| Disease–drug–protein reasoning | PrimeKG |

Nếu nghiên cứu tập trung vào **Biomedical KGQA hoặc multi-hop
reasoning**, có thể bắt đầu với PrimeKG vì graph có nhiều loại entity và
relation, đồng thời có tài liệu mô tả rõ mục tiêu precision medicine.

Nếu cần nghiên cứu entity normalization, semantic integration và
cross-species data, MonarchKG là lựa chọn đáng tìm hiểu.

------------------------------------------------------------------------

## 10. References

1.  **OREGANO (2023):** [The OREGANO knowledge graph for computational
    drug repurposing](https://doi.org/10.1038/s41597-023-02757-0).
    *Scientific Data*.
2.  **CKG (2022):** [A knowledge graph to interpret clinical proteomics
    data](https://doi.org/10.1038/s41587-021-01145-6). *Nature
    Biotechnology*.
3.  **Monarch Initiative (2024):** [The Monarch Initiative in 2024: an
    analytic platform integrating phenotypes, genes and diseases across
    species](https://doi.org/10.1093/nar/gkad1082). *Nucleic Acids
    Research*.
4.  **PrimeKG (2023):** [Building a knowledge graph to enable precision
    medicine](https://doi.org/10.1038/s41597-023-01960-3). *Scientific
    Data*.

### Ghi chú khi sử dụng tài liệu này

-   Các số liệu về CKG và MonarchKG là số liệu tham khảo, có thể phụ
    thuộc snapshot và cách thống kê. Hãy xác minh trực tiếp từ bản dữ
    liệu dùng trong experiment trước khi đưa vào thesis/paper.
-   Với mỗi dataset tải về, nên lưu: URL, ngày tải, release/version,
    license, số nodes, số edges, schema và preprocessing steps.
-   Không nên coi mọi edge trong KG là quan hệ nhân quả. Cần phân biệt
    association, correlation, interaction, treatment và causal relation.
-   Các ví dụ câu hỏi trong tài liệu là minh họa; cần kiểm tra schema và
    evidence của KG trước khi xem chúng là các fact đã được dataset xác
    nhận.
