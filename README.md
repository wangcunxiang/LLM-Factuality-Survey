# LLM-Factuality-Survey
The repository for the survey paper "Survey on Factuality in Large Language Models: Knowledge, Retrieval and Domain-Specificity"

![](survey-tree.jpg)

# Tables


*Table: Comparison between the factuality issue and the hallucination issue.*


| | |
|---|---|
|**Factual and Non-Hallucinated**|Factually correct outputs.|
|**Non-Factual and Hallucinated**|Entirely fabricated outputs.|  
|**Hallucinated but Factual**<br>  |1. Outputs that are unfaithful to the prompt but remain factually correct (cao-etal-2022-hallucinated).<br> 2. Outputs that deviate from the prompt's specifics but don't touch on factuality, e.g., a prompt asking for a story about a rabbit and wolf becoming friends, but the LLM produces a tale about a rabbit and a dog befriending each other.<br> 3. Outputs that provide additional factual details not specified in the prompt, e.g., a prompt asking about the capital of France, and the LLM responds with "Paris, which is known for the Eiffel Tower."|
|**Non-Factual but Non-Hallucinated**<br> |1. Outputs where the LLM states, "I don't know," or avoids a direct answer.<br> 2. Outputs that are partially correct, e.g., for the question, "Who landed on the moon with Apollo 11?" If the LLM responds with just "Neil Armstrong," the answer is incomplete but not hallucinated.<br> 3. Outputs that provide a generalized or vague response without specific details, e.g., for a question about the causes of World War II, the LLM might respond with "It was due to various political and economic factors."|

## Causes of Factual Errors
| Category | Cause | Example Dialog | Notes and references |
| --- | --- | --- | --- |
| Model-level causes | Domain knowledge deficit | **Q:** CEO of Assicurazioni Generali? <br> **BloombergGPT:** Philippe Donnet <br> **GPT-NeoX:** Antonio De Lorenzo, Simone Gambarini, Enrico Zanetti <br> **FLAN-T5-XXL:** John M Forsyth, Christopher K Peters, {empty string} | BloombergGPT is a finance domain-specific language model. [Wu et al, 2023](https://arxiv.org/abs/2303.17564) |
| | Outdated information | **Q:** When was Kyiv attacked by Russia?<br> **ChatGPT:**As of my last knowledge update in September 2021, Russia had not launched an attack on Kyiv. | Kyiv was attacked by Russia on 25 February 2022. |
| | Reasoning error | **Q:** Who is Tom Cruise’s mother? <br> **A:** Mary Lee Pfeiffer <br> **Q:** Who is Mary Lee Pfeiffer’s son? <br> **A:** There is no widely known information about... | From [Berglund et al, 2023](https://arxiv.org/abs/2309.12288). It is clear that the model knows Tom Cruise’s mother is Lee Pfeiffer, but it fails to reason that Lee Pfeiffer has a son named Tom Cruise. |
| Retrieval-level causes | Distracted by the retrieval information | **Retrieved context:** "Alan Melikdjanian - Age, Bio, Faces and Birthday"<br> Alan Melikdjanian is the Director, Writer (with his father), Editor, and Composer of the 2006 direct-to-DVD Russian-language film Citizen Mavzik...<br> **Q:** What is the nationality of the director of Citizen Mavzik? <br> **A:** (1) The director of the film Citizen Mavzik is Alan Melikdjanian. (2) Alan Melikdjanian is from Russia. So, the answer is Russia. |  Alan Melikdjanian, known by the alias Captain Disillusion, is a Soviet-born American independent filmmaker and YouTuber. [Melikdjanian et al, 2018](https://www.reddit.com/r/Damnthatsinteresting/comments/w9gauc/youtuber_captain_disillusion_explains_how_his/) |
| | Misunderstood the retrieval information | **Retrieved context:** "Gerard III, Count of Guelders - Wikipedia"<br> Gerard married Margaretha of Brabant...<br> **Q:** Who is Gerard I, Count Of Guelders's wife? <br> **A:** ... the name of Gerard I, Count of Guelders's wife, which is Margaretha of Brabant.| The retrieved context accurately provided the name of Gerard III, Count of Guelders' wife, which is Margaretha of Brabant. However, the model erroneously referred to him as Gerard I, resulting in an incorrect answer. |
| | Fail to address misinformation in retrieved documents | **Retrieved context:** [0] "James Vincent - Death Notice and Service Information - Legacy.com"<br> James Vincent died on August 9, 2023...<br> [1] "UPDATE: Jury convicts James Vincent of murder after deliberating..." A jury convicted James Vincent of murder in the 2018 Shooting death of Luka Grabb...<br> **Q:** When did James Vincent die? <br> **A:** James Vincent died on August 9, 2023, in Long Beach, California.| In the presence of contradictory information from retrieved documents, the model should thoroughly outline the response, while also noting the need for clarity. For instance, the model could state: "According to the first document, James Vincent passed away on August 9, 2023, in Long Beach. However, the second document suggests that James Vincent was murdered in 2018." |
| Inference-level causes | Snowballing | **Q:** Was there ever a US senator who represented the state of New Hampshire and whose alma mater was the University of Pennsylvania? <br> **A:** Yes... His name was John P. Hale | John P. Hale was graduated from Bowdoin College. [Zhang et al, 2023](https://arxiv.org/abs/2305.13534) |
| | Exposure bias | **Q:** Aamari was very stoic. [PRONOUN] rarely showed any emotion. <br> **A:** He. | The correct answer was Xe according to [Hossain et al, 2023](https://doi.org/10.18653/v1/2023.acl-long.293).
## Evaluations 

| Reference | Task | Dataset | Metrics | Human Eval | Evaluated LLMs | Granularity |
|-----------|------|---------|---------|------------|----------------|-------------|
| FActScore [ Min et al,  2023 ](https://arxiv.org/abs/2305.14251) | Biography Generation | 183 people entities | F1 | ✓ | GPT-3.5,<br/>ChatGPT... | T |
| SelfCheckGPT [Manakul et al, 2023](https://arxiv.org/abs/2303.08896) | Bio Generation | WikiBio | AUC-PR,<br/>Human Score | ✓ | GPT-3,<br/>LLaMA,<br/>OPT,<br/>GPT-J... | S |
| [Wang et al, 2023](https://arxiv.org/abs/2305.12421) | Open QA | NQ, TQ | ACC,<br/>EM | ✓ | GPT-3.5,<br/>ChatGPT,<br/>GPT-4,<br/>Bing Chat | S |
| [Pezeshkpour et al, 2023](https://arxiv.org/abs/2306.06264) | Knowledge Probing | T-REx,<br/>LAMA | ACC |  | GPT3.5 | T |
| [De Cao et al, 2021](https://doi.org/10.18653/v1/2021.emnlp-main.522) | QA,<br/>Fact Checking | KILT,<br/>FEVER,<br/>zsRE | ACC |  | GPT-3,<br/>FLAN-T5 | S/T |
| [Varshney et al, 2023](https://arxiv.org/abs/2307.03987) | Article Generation | Unnamed Dataset | ACC,<br/>AUC |  | GPT3.5,<br/>Vicuna | S |
| FactTool [Chern et al, 2023](https://arxiv.org/abs/2307.13528) | KB-based QA | RoSE | ACC,<br/>F1... |  | GPT-4,<br/>ChatGPT,<br/>FLAN-T5 | S |
| [Kadavath et al, 2022](https://arxiv.org/abs/2207.05221) | Self-evaluation | BIG Bench,<br/>MMLU, LogiQA,<br/>TruthfulQA,<br/>QuALITY, TriviaQA Lambada | ACC,<br/>Brier Score,<br/>RMS Calibration Error... |  | Claude | T |


| Reference | Task | Dataset | Metrics | Human Eval | Evaluated LLMs | Granularity |
| --------- | ---- | ------- | ------- | ---------- | -------------- | ----------- |
| Retro [Borgeaud et al, 2022](https://arxiv.org/abs/2112.04426) | QA,<br>Language<br>Modeling | MassiveText,<br> Curation Corpus,<br> Wikitext103,<br> Lambada,<br> C4,Pile, NQ | PPL,<br> ACC,<br> Exact Match | ✓ | Retro | T |
| GenRead [Yu et al, 2023](https://arxiv.org/abs/2209.10063) | QA,<br> Dialogue,<br> Fact Checking | NQ, TQ, WebQ,<br> FEVER,<br> FM2, WoW | EM, ACC,<br> F1, Rouge-L | - | GPT3.5, Codex<br>GPT-3, Gopher<br>FLAN, GLaM<br>PaLM | S |
| GopherCite [Menick et al, 2022](https://arxiv.org/abs/2203.11147) | Self-supported QA | NQ, ELI5,<br> TruthfulQA<br>(Health, Law, Fiction, Conspiracies) | Human Score | ✓ | GopherCite | S |
| Trivedi et al. [Trivedi et al, 2023](https://doi.org/10.18653/v1/2023.acl-long.557) | QA | HotpotQA, IIRC<br>2WikiMultihopQA,<br> MuSiQue(music) | Retrieval recall,<br> Answer F1 | - | GPT-3<br>FLAN-T5 | S/T |
| Peng et al. [Peng et al, 2023](https://arxiv.org/abs/2302.12813) | QA,<br> Dialogue | DSTC7 track2<br> DSTC11 track5,<br> OTT-QA | ROUGE, chrF,<br> BERTScore, Usefulness,<br> Humanness... | ✓ | ChatGPT | S/T |
| CRITIC [Gou et al, 2023](https://arxiv.org/abs/2305.11738) | QA<br>Toxicity Reduction | AmbigNQ, TriviaQA, HotpotQA,<br> RealToxicityPrompts | Exact Match, maximum toxicity,<br> perplexity, n-gram diversity,<br> AUROC..., | - | GPT-3.5<br>ChatGPT | T |
| Khot et al. [Khot et al, 2023](https://arxiv.org/abs/2210.02406) | QA,<br> long-context QA | CommaQA-E, 2WikiMultihopQA, MuSiQue, HotpotQA | Exact Match, Answer F1 | - | GPT-3<br>FLAN-T5 | T |
| ReAct [Yao et al, 2023](https://arxiv.org/abs/2210.03629) | QA<br> Fact Verification | HotpotQA, FEVER | Exact Match, ACC | - | PaLM<br>GPT-3 | S/T |
| Jiang et al. [Jiang et al, 2023](https://arxiv.org/abs/2305.06983) | QA, Commonsense Reasoning,<br> long-form QA... | 2WikiMultihopQA, StrategyQA, ASQA, WikiAsp | Exact Match, Disambig-F1, ROUGE,<br> entity F1... | - | GPT-3.5 | T |
| Lee et al. [Lee et al, 2022](https://arxiv.org/abs/2206.04624) | Open-ended Generation | FEVER | Entity score, Entailment<span>Ratio, ppl... | - | Megatron-LM | T |
| SAIL [Luo et al, 2023](https://arxiv.org/abs/2305.15225) | QA<br> Fact Checking | UniLC | ACC<br> F1 | - | LLaMA Vicuna<br>SAIL | T |
| He et al. [He et al, 2022](https://arxiv.org/abs/2301.00303) | Commonsense Reasoning, Temporal Reasoning,<br> Tabular Reasoning | StrategyQA, TempQuestions, IN-FOTABS | ACC | - | GPT-3 | T |
| Pan et al. [Pan et al, 2023](https://doi.org/10.18653/v1/2023.acl-long.386) | Fact Checking | HOVER<br> FEVEROUS-S | Macro-F1 | - | Codex<br>FLAN-T5 | S |
| Multiagent Debate [Du et al, 2023](https://arxiv.org/abs/2305.14325) | Biography<br> MMLU | Unnamed Biography Dataset,<br> MMLU | ChatGPT Evaluator, ACC | - | Bard<br>ChatGPT | S |

### Benchmarks

| Reference | Task Type | Dataset | Metrics | Performance of Representative LLMs |
|-----------|-----------|---------|---------|-----------------------------------|
| MMLU [Hendrycks et al, 2021](https://arxiv.org/abs/2009.03300) | Multi-Choice QA | Humanities,<br/>Social,<br/>Sciences,<br/>STEM... | ACC | (ACC, 5-shot)<br/>GPT-4: 86.4<br/>GPT-3.5: 70<br/>LLaMA2-70B: 68.9 |
| TruthfulQA [Lin et al, 2022](https://doi.org/10.18653/v1/2022.acl-long.229) | QA | Health, Law,<br/>Conspiracies,<br/>Fiction... | Human Score,<br/> GPT-judge, <br/> ROUGE, BLEU, <br/>MC1,MC2... | (zero-shot)<br/>GPT-4: ~29 (MC1)<br/>GPT-3.5: ~28 (MC1),<br/> 79.92(%true)<br/>LLaMA2-70B: 53.37 (%true) |
| C-Eval [Huang et al, 2023](https://arxiv.org/abs/2305.08322) | Multi-Choice QA | STEM,<br/>Social Science,<br/>Humanities... | ACC | (zero-shot, average ACC)<br/>GPT-4: 68.7<br/>GPT-3.5: 54.4<br/>LLaMA2-70B: 50.13 |
| AGIEval [Huang et al, 2023](https://arxiv.org/abs/2305.08322) | Multi-Choice QA | Gaokao, (geometry, Bio,<br/>history...),SAT, Law... | ACC | (zero-shot, average ACC)<br/>GPT-4: 56.4<br/>GPT-3.5: 42.9<br/>LLaMA2-70B: 40.02 |
| HaluEval [Li et al, 2023](https://arxiv.org/abs/2305.11747) | Hallucination Evaluation | HaluEval | ACC | (general ACC)<br/>GPT-3.5: 86.22 |
| BigBench [Srivastava et al, 2023](https://openreview.net/forum?id=uyTL5Bvosj) | Multi-tasks(QA, NLI, Fact Checking, Reasoning...) | BigBench | Metric to each type of task | (Big-Bench Hard)<br/>GPT-3.5: 49.6<br/>LLaMA-65B: 42.6 |
| ALCE [Gao et al, 2023](https://arxiv.org/abs/2305.14627) | Citation Generation | ASQA, ELI5,<br/>QAMPARI | MAUVE, Exact Match, ROUGE-L... | (ASQA, 3-psg, citation prec)<br/>GPT-3.5: 73.9<br/>LLaMA-33B: 23.0 |
| QUIP [Weller et al, 2023](https://arxiv.org/abs/2305.13252) | Generative QA | TriviaQA,<br/> NQ, ELI5,<br/>HotpotQA | QUIP-Score, Exact match | (ELI5, QUIP, null prompt)<br/>GPT-4: 21.0<br/>GPT-3.5: 27.7 |
| PopQA [Mallen et al, 2023](https://doi.org/10.18653/v1/2023.acl-long.546) | Multi-Choice QA | PopQA,<br/>EntityQuestions | ACC | (overall ACC)<br/>GPT-3.5: ~37.0 |
| UniLC [Zhang et al, 2023](https://arxiv.org/abs/2304.03728) | Fact Checking | Climate,<br/>Health, MGFN | ACC, F1 | (zero-shot, fact tasks, average F1)<br/>GPT-3.5: 51.62 |
| Pinocchio [Hu et al, 2023](https://arxiv.org/abs/2310.05177) | Fact Checking, QA, Reasoning | Pinocchio | ACC, F1 | GPT-3.5: (Zero-shot ACC: 46.8, F1:44.4)<br/>GPT-3.5: (Few-shot ACC: 47.1, F1:45.7) |
| SelfAware [Yin et al, 2023](https://doi.org/10.18653/v1/2023.findings-acl.551) | Self-evaluation | SelfAware | ACC | (instruction input, F1)<br/>GPT-4: 75.47<br/>GPT-3.5: 51.43<br/>LLaMA-65B: 46.89 |
| RealTimeQA [Kasai et al, 2022](https://arxiv.org/abs/2207.13332) | Multi-Choice QA, Generative QA | RealTimeQA | ACC, F1 | (original setting, GCS retrieval)<br/>GPT-3: 69.3 (ACC for MC)<br/>GPT-3: 39.4 (F1 for generation) |
| FreshQA [Vu et al, 2023](https://arxiv.org/abs/2310.03214) | Generative QA | FRESHQA | ACC (Human) | (strict ACC, null prompt)<br/>GPT-4: 28.6<br/>GPT-3.5: 26.0 |

## Domain evaluation

| Reference | Domain | Task | Datasets | Metrics | Evaluated LLMs |
|-----------|--------|------|----------|---------|----------------|
| [Xie et al, 2023](https://arxiv.org/abs/2306.05443) | Finance | Sentiment analysis,<br/> News headline classification,<br/> Named entity recognition,<br/> Question answering,<br/> Stock movement prediction | FLARE | F1, Acc,<br/> Avg F1,<br/> Entity F1,<br/> EM, MCC | GPT-4 ,<br/> BloombergGPT,<br/> FinMA-(7B, 30B, 7B-full),<br/> Vicuna-7B |
| [Li et al, 2023](https://arxiv.org/abs/2308.06966) | Finance | 134 E-com tasks | EcomInstruct | Micro-F1,<br/> Macro-F1,<br/> ROUGE | BLOOM, BLOOMZ,<br/> ChatGPT, EcomGPT|
| [Wang et al, 2023](https://arxiv.org/abs/2308.08833) | Medicine | Multi-Choice QA | CMB | Acc | GPT-4, ChatGLM2-6B,<br/> ChatGPT, DoctorGLM,<br/> Baichuan-13B-chat,<br/> HuatuoGPT, MedicalGPT,<br/> ChatMed-Consult,<br/> ChatGLM-Med ,<br/> Bentsao, BianQue-2|
| [Li et al, 2023](https://arxiv.org/abs/2305.01526) | Medicine | Generative-QA | Huatuo-26M | BLEU,<br/> ROUGE,<br/> GLEU | T5, GPT2 |
| [Jin et al, 2023](https://arxiv.org/abs/2304.09667) | Medicine | Nomenclature,<br/> Genomic location,<br/> Functional analysis,<br/> Sequence alignment | GeneTuring | Acc | GPT-2, BioGPT, <br/> BioMedLM, <br/> GPT-3, <br/> ChatGPT, New Bing|
| [Guha et al, 2023](https://arxiv.org/abs/2308.11462) | Law | Issue-spotting,<br/> Rule-recall,<br/> Rule-application,<br/> Rule-conclusion,<br/> Interpretation,<br/> Rhetorical-understanding | LegalBench | Acc, EM | GPT-4, GPT-3.5, <br/> Claude-1, Incite, OPT<br/> Falcon, LLaMA-2, FLAN-T5...|
| [Fei et al, 2023](https://arxiv.org/abs/2309.16289) | Law | Legal QA, NER, <br/> Sentiment Analysis,<br/> Reading Comprehension | LawBench | F1, Acc,<br/> ROUGE-L,<br/> Normalized log-distance... | GPT-4,<br/> ChatGPT, <br/> InternLM-Chat,<br/> StableBeluga2...|

## Enhancement

### Enhancement methods

| Reference | Dataset | Metrics | Baselines ➝ Theirs | Dataset | Metrics | Baselines ➝ Theirs |
| --- | --- | --- | --- | --- | --- | --- |
| [Li et al, 2022](https://openreview.net/forum?id=02dbnEbEFn) | NQ | EM | 34.5 ➝ 44.35 (T5 11B) | GSM8K | ACC | 77.0 ➝ 85.0 (ChatGPT) |
| [Yu et al, 2023](https://arxiv.org/abs/2209.10063) | NQ | EM | 20.9 ➝ 28.0 (InstructGPT) | TriviaQA | EM | 57.5 ➝ 59.0 (InstructGPT) |
| - | - | - | - | WebQA | EM | 18.6 ➝ 24.6 (InstructGPT) |
| [Chuang et al, 2023](https://arxiv.org/abs/2309.03883) | FACTOR News | ACC | 58.3 ➝ 62.0 (LLaMa-7B) | FACTOR News | ACC | 61.1 ➝ 62.5 (LLaMa-13B) |
| - | FACTOR News | ACC | 63.8 ➝ 65.4 (LLaMa-33B) | FACTOR News | ACC | 63.6 ➝ 66.2 (LLaMa-65B) |
| - | FACTOR Wiki  | ACC | 58.6 ➝ 62.2 (LLaMa-7B) | FACTOR Wiki  | ACC | 62.6 ➝ 66.2 (LLaMa-13B) |
| - | FACTOR Wiki  | ACC | 69.5 ➝ 70.3 (LLaMa-33B) | FACTOR Wiki  | ACC | 72.2 ➝ 72.4 (LLaMa-65B) |
| - | TruthfulQA | %Truth * Info | 32.4 ➝ 44.6 (LLaMa-13B) | TruthfulQA | %Truth * Info | 34.8 ➝ 49.2 (LLaMa-65B) |
| [Li et al, 2022](https://arxiv.org/abs/2210.15097) | TruthfulQA | %Truth * Info | 32.4 ➝ 44.4 (LLaMa-13B) | TruthfulQA | %Truth * Info | 31.7 ➝ 36.7 (LLaMa-33B) |
| - | TruthfulQA | %Truth * Info | 34.8 ➝ 43.4 (LLaMa-65B) | - | - | - |
| [Li et al, 2023](https://arxiv.org/abs/2306.03341) | NQ | ACC | 46.6 ➝ 51.3 (LLaMA-7B) | TriviaQA | ACC | 89.6 ➝ 91.1 (LLaMA-7B) |
| - | MMLU | ACC  | 35.7 ➝ 40.1 (LLaMA-7B) | TruthfulQA | %Truth * Info | 32.5 ➝ 65.1 (Alpaca) |
| - |  TruthfulQA | %Truth * Info | 26.9 ➝ 43.5 (LLaMa-7B) |  TruthfulQA | %Truth * Info | 51.5 ➝ 74.0 (Vicuna) |
| [Cohen et al, 2023](https://arxiv.org/abs/2305.13281) | LAMA | F1 | 50.7 ➝ 80.8 (ChatGPT) | TriviaQA | F1 | 56.2 ➝ 82.6 (ChatGPT) |
| - | NQ | F1 | 60.6 ➝ 79.1 (ChatGPT) | PopQA | F1 | 65.2 ➝ 85.4 (ChatGPT) |
| - | LAMA | F1 | 42.5 ➝ 79.3 (GPT-3) | TriviaQA | F1 | 46.7 ➝ 77.2 (GPT-3) |
| - | NQ | F1 | 52.0 ➝ 78.0 (GPT-3) | PopQA | F1 | 43.7 ➝ 77.4 (GPT-3) |
|...|

### Domain-enhanced LLMs

| Reference | Domain | Model | Eval Task | Eval Dataset | Continual Pretrained? | Continual SFT? | Train From Scratch? | External Knowledge |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [Zhang et al, 2023](https://arxiv.org/abs/2305.15075) | Healthcare | Baichuan-7B, Ziya-LLaMA-13B | QA | cMedQA2, WebMedQA, Huatuo-26M | ✔️ | | | |
| [Yang et al, 2023](https://arxiv.org/abs/2308.03549) | Healthcare | Ziya-LLaMA-13B | QA | CMtMedQA, huatuo-26M | ✔️ | ✔️ | | |
| [Wang et al, 2023](https://arxiv.org/abs/2309.02233) | Healthcare | GPT-3.5-Turbo, LLaMA-2-13B | QA | MedQAUSMLE, MedQAMCMLE, MedMCQA | | | | ✔️ |
| [Ross et al, 2022](https://arxiv.org/abs/2106.09553) | Healthcare | MOLFORMER | Molecule properties prediction | | | | ✔️ | |
| [Bao et al, 2023](https://arxiv.org/abs/2308.14346) | Healthcare | Baichuan-13B | QA | CMB-Clin, CMD, CMID | | ✔️ | | |
| [Guan et al, 2023](https://arxiv.org/abs/2307.11346) | Healthcare | ChatGPT | IU-RR, MIMIC-CXR | | | | | ✔️ |
| [Liu et al, 2023](https://arxiv.org/abs/2303.11032) | Healthcare | GPT-4 | Medical Text De-Identification | | | | | ✔️ |
| [Li et al, 2023](https://arxiv.org/abs/2303.14070) | Healthcare | LLaMA | QA | | | ✔️ | | |
| [Venigalla et al, 2022](https://www.mosaicml.com/blog/introducing-pubmed-gpt) | Healthcare | GPT (2.7b) | QA |  | | | ✔️ | |
| [Xiong et al, 2023](https://arxiv.org/abs/2304.01097) | Healthcare | ChatGLM-6B | QA | | | ✔️ | | |
| [Tan et al, 2023](https://arxiv.org/abs/2309.01114) | Healthcare | Baichuan-7B | QA | C-Eval, MMLU | | ✔️ | | |
| [Luo et al, 2022](https://arxiv.org/abs/2210.10341) | Healthcare | GPT-2 | QA, DC, RE | | | | ✔️ | |
| [Jin et al, 2023](https://arxiv.org/abs/2304.09667) | Healthcare | Codex | QA | GeneTuring | | | | ✔️ |
| [Zakka et al, 2023](https://arxiv.org/abs/2303.01229) | Healthcare | text-davinci-003 | QA | ClinicalQA | | | | ✔️ |
| [Liu et al, 2023](https://arxiv.org/abs/2305.10688) | Healthcare | GPT-2medium | Molecular Property Prediction, Molecule-text translation | | | ✔️ | ✔️ | |
| [Nguyen et al, 2023](https://arxiv.org/abs/2302.05729) | Law | GPT3 | | | | ✔️ | | |
| [Savelka et al, 2023](https://arxiv.org/abs/2306.09525) | Law | GPT-4 | | | | | | ✔️ |
| [Huang et al, 2023](https://arxiv.org/abs/2305.15062) | Law | LLaMA | CN Legal Tasks | | ✔️ | ✔️ | | |
| [Cui et al, 2023](https://arxiv.org/abs/2306.16092) | Law | Ziya-LLaMA-13B | QA | national judicial examination question | ✔️ | | | ✔️ |
| [Li et al, 2023](https://arxiv.org/abs/2308.06966) | Finance | BLOOMZ | 4 major tasks 12 subtasks | EcomInstruct | | ✔️ | | |
| [Wu et al, 2023](https://arxiv.org/abs/2303.17564) | Finance | BLOOM | Financial NLP (SA, BC, NER, NER+NED, QA) | Financial Datasets | | | ✔️ | |
| [Deng et al, 2023](https://arxiv.org/abs/2306.05064) | Geoscience | LLaMA-7B | | GeoBench | ✔️ | | | |
| [Bai et al, 2023](https://arxiv.org/abs/2308.01414) | Geoscience | ChatGLM-6B | | | ✔️ | | | |
| [Fan et al, 2023](https://arxiv.org/abs/2307.13923) | E | phoenix-inst-chat-7b | Chinese Grammatical Error Correction | ChatGPT-generated, Human-annotated | | ✔️ | | |
| [Qi et al, 2023](https://arxiv.org/abs/2308.10173) | Food | Chinese-LLaMA2-13B | QA | | ✔️ | | | ✔️ |
| [Wen et al, 2023](https://arxiv.org/abs/2307.15290) | Home Renovation | Baichuan-13B | | C-Eval, CMMLU, EvalHome | | ✔️ | | |


# Paper List

## Factuality Enhancement

### On Standalone LLM Generation

#### Pretraining-based

##### Initial Pretraining

1. A Multitask, Multilingual, Multimodal Evaluation of ChatGPT on Reasoning, Hallucination, and Interactivity. _Yejin Bang et al._ arXiv 2023. [[paper](https://arxiv.org/abs/2302.04023)]
2. Deduplicating Training Data Makes Language Models Better. _Lee, Katherine et al._ ACL 2022. [[paper](https://aclanthology.org/2022.acl-long.577)]
3. Unsupervised Improvement of Factual Knowledge in Language Models. _Sadeq, Nafis et al._ EACL 2023. [[paper](https://aclanthology.org/2023.eacl-main.215.pdf)]

##### Continual Pretraining

1.   Factuality Enhanced Language Models for Open-Ended Text Generation. _Lee, Nayeon et al._ NIPS 2022. [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/df438caa36714f69277daa92d608dd63-Paper-Conference.pdf)]

#### Supervised Finetuning

##### Continual SFT

1.   SKILL: Structured Knowledge Infusion for Large Language Models. _Moiseev, Fedor et al._ NAACL 2022. [[paper](https://aclanthology.org/2022.naacl-main.113)]
2.   Contrastive Learning Reduces Hallucination in Conversations. _Sun, Weiwei et al._ AAAI 2023. [[paper](https://arxiv.org/pdf/2212.10400.pdf)]
3.   ChatGPT is not Enough: Enhancing Large Language Models with Knowledge Graphs for Fact-aware Language Modeling. _Linyao Yang et al._ arXiv 2023. [[paper](https://arxiv.org/pdf/2306.11489.pdf)]

##### Model Editing

1.   Editing Large Language Models: Problems, Methods, and Opportunities. _Yunzhi Yao et al._ arXiv 2023. [[paper](https://arxiv.org/pdf/2305.13172.pdf)]
2.   Knowledge Neurons in Pretrained Transformers. _Dai, Damai et al._ ACL 2022. [[paper](https://aclanthology.org/2022.acl-long.581.pdf)]
3.   Locating and Editing Factual Associations in GPT. _Kevin Meng et al._ NIPS 2022. [[paper](https://openreview.net/forum?id=-h6WAS6eE4)]
4.   Editing Factual Knowledge in Language Models. _De Cao, Nicola et al._ EMNLP 2021. [[paper](https://aclanthology.org/2021.emnlp-main.522.pdf)]
5.   Fast Model Editing at Scale. _Eric Mitchell et al._ ICLR 2022. [[paper](https://openreview.net/forum?id=0DcZxeWfOPt)]
6.   Inference-Time Intervention: Eliciting Truthful Answers from a Language Model. _Kenneth Li et al._ arXiv 2023. [[paper](https://arxiv.org/pdf/2306.03341.pdf)]

#### Multi-Agent

1.   Improving Factuality and Reasoning in Language Models through Multiagent Debate. _Yilun Du et al._ arXiv 2023. [[paper](https://arxiv.org/pdf/2305.14325.pdf)]
2.   LM vs LM: Detecting Factual Errors via Cross Examination. _Roi Cohen et al._ arXiv 2023. [[paper](https://arxiv.org/pdf/2305.13281.pdf)]

#### Novel Prompt

1.   Generate Rather than Retrieve: Large Language Models are Strong Context Generators. _Yu, Wenhao et al._  ICLR 2023. [[paper](https://arxiv.org/pdf/2209.10063.pdf)]
2.   "According to ..." Prompting Language Models Improves Quoting from Pre-Training Data. _Orion Weller et al._ arXiv 2023. [[paper](https://arxiv.org/pdf/2305.13252.pdf)]
3.   Decomposed Prompting: A Modular Approach for Solving Complex Tasks. _Tushar Khot et al._ arXiv 2023. [[paper](https://arxiv.org/pdf/2210.02406.pdf)]
4.   Chain-of-Verification Reduces Hallucination in Large Language Models. _Dhuliawala et al._ arXiv 2023. [[paper](https://arxiv.org/pdf/2309.11495.pdf)]

#### Decoding

1.   Factuality Enhanced Language Models for Open-Ended Text Generation. _Lee, Nayeon et al._ NIPS 2022. [[paper]([df438caa36714f69277daa92d608dd63-Paper-Conference.pdf (neurips.cc)](https://proceedings.neurips.cc/paper_files/paper/2022/file/df438caa36714f69277daa92d608dd63-Paper-Conference.pdf))]
2.   DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models. _Chuang, Yung-Sung et al._ arXiv 2023. [[paper](https://arxiv.org/pdf/2309.03883.pdf)]

 ### On Retrieval-Augmented Generation

#### Normal RAG Setting

1.   Improving Language Models by Retrieving From Trillions of Tokens. _Sebastian Borgeaud et al._ arXiv 2021. [[paper](https://arxiv.org/pdf/2112.04426.pdf)]
2.   Internet-Augmented Language Models through Few-Shot Prompting for Open-Domain Question Answering. _Angeliki Lazaridou et al._ arXiv 2022. [[paper](https://arxiv.org/pdf/2203.05115.pdf)]

#### Interactive Retrieval

##### CoT-based Retrieval

1.   Rethinking with Retrieval: Faithful Large Language Model Inference. _Hangfeng He et al._ arXiv 2023. [[paper](https://arxiv.org/pdf/2301.00303.pdf)]
2.   Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions. _Trivedi, Harsh et al._ ACL 2023. [[paper](https://aclanthology.org/2023.acl-long.557.pdf)]
3.   Active Retrieval Augmented Generation. _Zhengbao Jiang et al._ arXiv 2023. [[paper](https://arxiv.org/pdf/2305.06983.pdf)]

##### Agent-based Retrieval

1.   ReAct: Synergizing Reasoning and Acting in Language Models. _Shunyu Yao et al._  arXiv 2023. [[paper](https://arxiv.org/pdf/2210.03629.pdf)]
2.   Reflexion: Language Agents with Verbal Reinforcement Learning. _Noah Shinn et al._ arXiv 2023. [[paper](https://arxiv.org/pdf/2303.11366.pdf)]
3.   A Stitch in Time Saves Nine: Detecting and Mitigating Hallucinations of LLMs by Validating Low-Confidence Generation. _Neeraj Varshney et al._ arXiv 2023. [[paper](https://arxiv.org/pdf/2307.03987.pdf)]

#### Retrieval Adaptation

##### Prompt-based

1.   Check Your Facts and Try Again: Improving Large Language Models with External Knowledge and Automated Feedback. _Baolin Peng et al._ arXiv 2023. [[paper](https://arxiv.org/pdf/2302.12813.pdf)]

##### SFT-based

1.   Atlas: Few-shot Learning with Retrieval Augmented Language Models. _Gautier Izacard et al._ arXiv 2022. [[paper](https://arxiv.org/pdf/2208.03299.pdf)]
2.   REPLUG: Retrieval-Augmented Black-Box Language Models. _Weijia Shi et al._ arXiv 2023. [[paper](https://arxiv.org/pdf/2301.12652.pdf)]
3.   SAIL: Search-Augmented Instruction Learning. _Luo, Hongyin et al._ arXiv 2023. [[paper](https://arxiv.org/pdf/2305.15225.pdf)]

##### RLHF-based

1.   Teaching Language Models to Support Answers with Verified Quotes. _Jacob Menick et al._ arXiv 2022. [[paper](https://arxiv.org/pdf/2203.11147.pdf)]

#### Retrieval on External Memory

1.   Decoupled Context Processing for Context Augmented Language Modeling. _Zonglin Li et al._ NIPS 2022. [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/882d801fb1017f955547d5a816ade0fc-Paper-Conference.pdf)]
2.   G-MAP: General Memory-Augmented Pre-trained Language Model for Domain Tasks. _Zhongwei Wan et al._ ICML 2019. [[paper](http://proceedings.mlr.press/v97/houlsby19a/houlsby19a.pdf)]
3.    Parameter-Efficient Transfer Learning for NLP. _Neil Houlsby et al._ EMNLP 2022. [[paper](https://aclanthology.org/2022.emnlp-main.441.pdf)]
4.   KALA: Knowledge-Augmented Language Model Adaptation. _Kang, Minki et al._ NAACL 2022. [[paper](https://aclanthology.org/2022.naacl-main.379.pdf)]
5.   Entities as Experts: Sparse Memory Access with Entity Supervision. _Thibault Févry et al._ EMNLP 2020. [[paper](https://aclanthology.org/2020.emnlp-main.400.pdf)]
6.   Mention Memory: Incorporating Textual Knowledge into Transformers through Entity Mention Attention. _Michiel de Jong et al._ ICLR 2022. [[paper](https://openreview.net/pdf?id=OY1A8ejQgEX)]
7.   Plug-and-Play Knowledge Injection for Pre-trained Language Models. _Zhang, Zhengyan et al._ ACL 2023. [[paper](https://aclanthology.org/2023.acl-long.594.pdf)]
8.   Evidence-based Factual Error Correction. _Thorne, James et al._ ACL 2021. [[paper](https://aclanthology.org/2021.acl-long.256.pdf)]
9.   Rarr: Researching and revising what language models say, using language models. _Gao, Luyu et al._ ACL 2023. [[paper](https://aclanthology.org/2023.acl-long.910.pdf)]
10.   PURR: Efficiently Editing Language Model Hallucinations by Denoising Language Model Corruptions. _Chen, Anthony et al._ arXiv 2023. [[paper](https://arxiv.org/pdf/2305.14908.pdf)]

#### Retrieval on Structured Knowledge Source

1.   Mitigating Language Model Hallucination with Interactive Question-Knowledge Alignment. _Shuo Zhang et al._ arXiv 2023. [[paper](https://arxiv.org/pdf/2305.13669.pdf)]
2.   StructGPT: A general framework for Large Language Model to Reason on Structured Data. _Jinhao Jiang et al._ arXiv 2023. [[paper](https://arxiv.org/pdf/2305.09645.pdf)]
