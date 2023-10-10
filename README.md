# LLM-Factuality-Survey
The repository for the survey paper "Large Language Models Factuality Survey - Knowledge, Retrieval and Domain-Specificity"


TODO: Fix references and hyperlinks.

# Factuality and Hallucination


*Table: Comparison between the factuality issue and the hallucination issue.*


| | |
|---|---|
|**Factual and Non-Hallucinated**|Factually correct outputs.|
|**Non-Factual and Hallucinated**|Entirely fabricated outputs.|  
|**Hallucinated but Factual**<br>  |1. Outputs that are unfaithful to the prompt but remain factually correct (cao-etal-2022-hallucinated).<br> 2. Outputs that deviate from the prompt's specifics but don't touch on factuality, e.g., a prompt asking for a story about a rabbit and wolf becoming friends, but the LLM produces a tale about a rabbit and a dog befriending each other.<br> 3. Outputs that provide additional factual details not specified in the prompt, e.g., a prompt asking about the capital of France, and the LLM responds with "Paris, which is known for the Eiffel Tower."|
|**Non-Factual but Non-Hallucinated**<br> |1. Outputs where the LLM states, "I don't know," or avoids a direct answer.<br> 2. Outputs that are partially correct, e.g., for the question, "Who landed on the moon with Apollo 11?" If the LLM responds with just "Neil Armstrong," the answer is incomplete but not hallucinated.<br> 3. Outputs that provide a generalized or vague response without specific details, e.g., for a question about the causes of World War II, the LLM might respond with "It was due to various political and economic factors."|

# Causes

| Category  | Cause  | Example Dialog  | Notes and references |
| --- | --- | --- | --- |
|   | Domain knowledge deficit  | Q: CEO of Assicurazioni Generali?  <br> **BloombergGPT:** Philippe Donnet <br> **GPT-NeoX:** Antonio De Lorenzo, Simone Gambarini, Enrico Zanetti  <br> **FLAN-T5-XXL:** John M Forsyth, Christopher K Peters, {empty string} | BloombergGPT is a finance domain-specific language model.[^1] |
|   | Outdated information  | Q: When was Kyiv attacked by Russia?  <br> **ChatGPT:** As of my last knowledge update in September 2021, Russia had not launched an attack on Kyiv. | Kyiv was attacked by Russia on 25 February 2022.[^2] |
| Model-level causes | Reasoning error | Q: Who is Tom Cruise’s mother? <br> A: Mary Lee Pfeiffer  <br> Q: Who is Mary Lee Pfeiffer’s son? <br> A: There is no widely known information about... | From [^3]. It is clear that the model knows Tom Cruise’s mother is Lee Pfeiffer, but it fails to reason that Lee Pfeiffer has a son named Tom Cruise. |
| Retrieval-level causes | Distracted by the retrieval information | **Retrieved context:** "Alan Melikdjanian - Age, Bio, Faces and Birthday".. <br> Q: What is the nationality of the director of Citizen Mavzik? <br> A: (1) The director of the film Citizen Mavzik is Alan Melikdjanian.  (2) Alan Melikdjanian is from Russia.  So, the answer is Russia. | Alan Melikdjanian, known by the alias Captain Disillusion, is a Soviet-born American independent filmmaker and YouTuber[^4]. |
|   | Misunderstood the retrieval information | **Retrieved context:** "Gerard III, Count of Guelders - Wikipedia"... <br> Q: What is the place of birth of Gerard I, Count Of Guelders's wife?  <br> A: ... the name of Gerard I, Count of Guelders's wife, which is Margaretha of Brabant. | The retrieved context does not provide information about her place of birth which was the original question. The model then incorrectly provides the name of Gerard I, Count of Guelders's wife, Margaretha of Brabant, instead of providing the place of birth.  |
|   | Fail to address misinformation in retrieved documents | **Retrieved context:** [0] "James Vincent - Death Notice and Service Information - Legacy.com".. <br> Q: When did James Vincent die? <br> A: Based on the information provided in the two articles, James Vincent died on August 9, 2023, in Long Beach, California. | In the presence of contradictory information from retrieved documents, the model should thoroughly outline the response, while also noting the need for clarity.[^5] |
| Inference-level causes | Snowballing | Q: Was there ever a US senator who represented the state of New Hampshire and whose alma mater was the University of Pennsylvania? <br> A: Yes... His name was John P. Hale | John P. Hale was graduated from Bowdoin College.[^6] |
|   | Exposure bias | Q: Aamari was very stoic. [PRONOUN] rarely showed any emotion. <br> A: He. | The correct answer was Xe according to [^7]. |

> Examples of different kinds of factual errors produced by large language models. We category the factual error types by the causes of them, whose details can be found in Sec[^8]. 

[^1]: BloombergGPT
[^2]: Kyiv attack
[^3]: berglund2023reversal
[^4]: Captain_disillusion_2018
[^5]: Misinformation resolve
[^6]: Snowball
[^7]: hossain-etal-2023-misgendered
[^8]: Section reference

# Evaluations 

| Reference | Task | Dataset | Metrics | Human Eval | Evaluated LLMs | Granularity |
|-----------|------|---------|---------|------------|----------------|-------------|
| FActScore \citep{Min2023-FActScore} | Biography Generation | 183 people entities | F1 | ✓ | GPT-3.5,<br/>ChatGPT... | T |
| SelfCheckGPT \citep{SelfCheckGPT} | Bio Generation | WikiBio | AUC-PR,<br/>Human Score | ✓ | GPT-3,<br/>LLaMA,<br/>OPT,<br/>GPT-J... | S |
| \citep{wang2023evaluating} | Open QA | NQ, TQ | ACC,<br/>EM | ✓ | GPT-3.5,<br/>ChatGPT,<br/>GPT-4,<br/>Bing Chat | S |
| \citep{pezeshkpour2023measuring} | Knowledge Probing | T-REx,<br/>LAMA | ACC |  | GPT3.5 | T |
| \citep{de-cao-etal-2021-editing} | QA,<br/>Fact Checking | KILT,<br/>FEVER,<br/>zsRE | ACC |  | GPT-3,<br/>FLAN-T5 | S/T |
| \citep{varshney2023stitch} | Article Generation | Unnamed Dataset | ACC,<br/>AUC |  | GPT3.5,<br/>Vicuna | S |
| FactTool \citep{chern2023factool} | KB-based QA | RoSE | ACC,<br/>F1... |  | GPT-4,<br/>ChatGPT,<br/>FLAN-T5 | S |
| \citep{kadavath2022language} | Self-evaluation | BIG Bench,<br/>MMLU, LogiQA,<br/>TruthfulQA,<br/>QuALITY, TriviaQA Lambada | ACC,<br/>Brier Score,<br/>RMS Calibration Error... |  | Claude | T |


| Reference | Task | Dataset | Metrics | Human Eval | Evaluated LLMs | Granularity |
| --------- | ---- | ------- | ------- | ---------- | -------------- | ----------- |
| Retro [[borgeaud2022improving]](https://github.com/YTsai28/ltmi) | QA,<br>Language<br>Modeling | MassiveText,<br> Curation Corpus,<br> Wikitext103,<br> Lambada,<br> C4,Pile, NQ | PPL,<br> ACC,<br> Exact Match | ✓ | Retro | T |
| GenRead [[yu2023generate]](https://github.com/YTsai28/ltmi) | QA,<br> Dialogue,<br> Fact Checking | NQ, TQ, WebQ,<br> FEVER,<br> FM2, WoW | EM, ACC,<br> F1, Rouge-L | - | GPT3.5, Codex<br>GPT-3, Gopher<br>FLAN, GLaM<br>PaLM | S |
| GopherCite [[menick2022teaching]](https://github.com/YTsai28/ltmi) | Self-supported QA | NQ, ELI5,<br> TruthfulQA<br>(Health, Law, Fiction, Conspiracies) | Human Score | ✓ | GopherCite | S |
| Trivedi et al. [[trivedi-etal-2023-interleaving]](https://github.com/YTsai28/ltmi) | QA | HotpotQA, IIRC<br>2WikiMultihopQA,<br> MuSiQue(music) | Retrieval recall,<br> Answer F1 | - | GPT-3<br>FLAN-T5 | S/T |
| Peng et al. [[peng2023check]](https://github.com/YTsai28/ltmi) | QA,<br> Dialogue | DSTC7 track2<br> DSTC11 track5,<br> OTT-QA | ROUGE, chrF,<br> BERTScore, Usefulness,<br> Humanness... | ✓ | ChatGPT | S/T |
| CRITIC [[gou2023critic]](https://github.com/YTsai28/ltmi) | QA<br>Toxicity Reduction | AmbigNQ, TriviaQA, HotpotQA,<br> RealToxicityPrompts | Exact Match, maximum toxicity,<br> perplexity, n-gram diversity,<br> AUROC..., | - | GPT-3.5<br>ChatGPT | T |
| Khot et al. [[khot2023decomposed]](https://github.com/YTsai28/ltmi) | QA,<br> long-context QA | CommaQA-E, 2WikiMultihopQA, MuSiQue, HotpotQA | Exact Match, Answer F1 | - | GPT-3<br>FLAN-T5 | T |
| ReAct [[yao2023react]](https://github.com/YTsai28/ltmi) | QA<br> Fact Verification | HotpotQA, FEVER | Exact Match, ACC | - | PaLM<br>GPT-3 | S/T |
| Jiang et al. [[jiang2023active]](https://github.com/YTsai28/ltmi) | QA, Commonsense Reasoning,<br> long-form QA... | 2WikiMultihopQA, StrategyQA, ASQA, WikiAsp | Exact Match, Disambig-F1, ROUGE,<br> entity F1... | - | GPT-3.5 | T |
| Lee et al. [[lee2022factuality]](https://github.com/YTsai28/ltmi) | Open-ended Generation | FEVER | Entity score, Entailment<span>Ratio, ppl... | - | Megatron-LM | T |
| SAIL [[luo2023sail]](https://github.com/YTsai28/ltmi) | QA<br> Fact Checking | UniLC | ACC<br> F1 | - | LLaMA Vicuna<br>SAIL | T |
| He et al. [[he2022rethinking]](https://github.com/YTsai28/ltmi) | Commonsense Reasoning, Temporal Reasoning,<br> Tabular Reasoning | StrategyQA, TempQuestions, IN-FOTABS | ACC | - | GPT-3 | T |
| Pan et al. [[pan-etal-2023-fact]](https://github.com/YTsai28/ltmi) | Fact Checking | HOVER<br> FEVEROUS-S | Macro-F1 | - | Codex<br>FLAN-T5 | S |
| Multiagent Debate [[multiagent_debate]](https://github.com/YTsai28/ltmi) | Biography<br> MMLU | Unnamed Biography Dataset,<br> MMLU | ChatGPT Evaluator, ACC | - | Bard<br>ChatGPT | S |


## Benchmarks

| Reference | Task Type | Dataset | Metrics | Performance of Representative LLMs |
|-----------|-----------|---------|---------|-----------------------------------|
| MMLU \citep{MMLU} | Multi-Choice QA | Humanities,<br/>Social,<br/>Sciences,<br/>STEM... | ACC | (ACC, 5-shot)<br/>GPT-4: 86.4<br/>GPT-3.5: 70<br/>LLaMA2-70B: 68.9 |
| TruthfulQA \citep{TruthfulQA} | QA | Health, Law,<br/>Conspiracies,<br/>Fiction... | Human Score,<br/> GPT-judge, <br/> ROUGE, BLEU, <br/>MC1,MC2... | (zero-shot)<br/>GPT-4: ~29 (MC1)<br/>GPT-3.5: ~28 (MC1),<br/> 79.92(%true)<br/>LLaMA2-70B: 53.37 (%true) |
| C-Eval \citep{C-Eval} | Multi-Choice QA | STEM,<br/>Social Science,<br/>Humanities... | ACC | (zero-shot, average ACC)<br/>GPT-4: 68.7<br/>GPT-3.5: 54.4<br/>LLaMA2-70B: 50.13 |
| AGIEval \citep{C-Eval} | Multi-Choice QA | Gaokao, (geometry, Bio,<br/>history...),SAT, Law... | ACC | (zero-shot, average ACC)<br/>GPT-4: 56.4<br/>GPT-3.5: 42.9<br/>LLaMA2-70B: 40.02 |
| HaluEval \citep{HaluEval} | Hallucination Evaluation | HaluEval | ACC | (general ACC)<br/>GPT-3.5: 86.22 |
| BigBench \citep{BigBench} | Multi-tasks(QA, NLI, Fact Checking, Reasoning...) | BigBench | Metric to each type of task | (Big-Bench Hard)<br/>GPT-3.5: 49.6<br/>LLaMA-65B: 42.6 |
| ALCE \citep{gao2023enabling} | Citation Generation | ASQA, ELI5,<br/>QAMPARI | MAUVE, Exact Match, ROUGE-L... | (ASQA, 3-psg, citation prec)<br/>GPT-3.5: 73.9<br/>LLaMA-33B: 23.0 |
| QUIP \citep{weller2023according} | Generative QA | TriviaQA,<br/> NQ, ELI5,<br/>HotpotQA | QUIP-Score, Exact match | (ELI5, QUIP, null prompt)<br/>GPT-4: 21.0<br/>GPT-3.5: 27.7 |
| PopQA \citep{PopQA} | Multi-Choice QA | PopQA,<br/>EntityQuestions | ACC | (overall ACC)<br/>GPT-3.5: ~37.0 |
| UniLC \citep{UniLC} | Fact Checking | Climate,<br/>Health, MGFN | ACC, F1 | (zero-shot, fact tasks, average F1)<br/>GPT-3.5: 51.62 |
| Pinocchio \citep{Pinocchio} | Fact Checking, QA, Reasoning | Pinocchio | ACC, F1 | GPT-3.5: (Zero-shot ACC: 46.8, F1:44.4)<br/>GPT-3.5: (Few-shot ACC: 47.1, F1:45.7) |
| SelfAware \citep{yin-etal-2023-large} | Self-evaluation | SelfAware | ACC | (instruction input, F1)<br/>GPT-4: 75.47<br/>GPT-3.5: 51.43<br/>LLaMA-65B: 46.89 |
| RealTimeQA \citep{kasai2022realtimeqa} | Multi-Choice QA, Generative QA | RealTimeQA | ACC, F1 | (original setting, GCS retrieval)<br/>GPT-3: 69.3 (ACC for MC)<br/>GPT-3: 39.4 (F1 for generation) |
| FreshQA \citep{vu2023freshllms} | Generative QA | FRESHQA | ACC (Human) | (strict ACC, null prompt)<br/>GPT-4: 28.6<br/>GPT-3.5: 26.0 |

## Domain evaluation

| Reference | Domain | Task | Datasets | Metrics | Evaluated LLMs |
|-----------|--------|------|----------|---------|----------------|
| \cite{xie2023pixiu} | Finance | Sentiment analysis,<br/> News headline classification,<br/> Named entity recognition,<br/> Question answering,<br/> Stock movement prediction | FLARE | F1, Acc,<br/> Avg F1,<br/> Entity F1,<br/> EM, MCC | GPT-4 ,<br/> BloombergGPT,<br/> FinMA-(7B, 30B, 7B-full),<br/> Vicuna-7B |
| \cite{li2023ecomgpt} | Finance | 134 E-com tasks | EcomInstruct | Micro-F1,<br/> Macro-F1,<br/> ROUGE | BLOOM, BLOOMZ,<br/> ChatGPT, EcomGPT|
| \cite{wang2023cmb} | Medicine | Multi-Choice QA | CMB | Acc | GPT-4, ChatGLM2-6B,<br/> ChatGPT, DoctorGLM,<br/> Baichuan-13B-chat,<br/> HuatuoGPT, MedicalGPT,<br/> ChatMed-Consult,<br/> ChatGLM-Med ,<br/> Bentsao, BianQue-2|
| \cite{li2023huatuo} | Medicine | Generative-QA | Huatuo-26M | BLEU,<br/> ROUGE,<br/> GLEU | T5, GPT2 |
| \cite{jin2023genegpt} | Medicine | Nomenclature,<br/> Genomic location,<br/> Functional analysis,<br/> Sequence alignment | GeneTuring | Acc | GPT-2, BioGPT, <br/> BioMedLM, <br/> GPT-3, <br/> ChatGPT, New Bing|
| \cite{guha2023legalbench} | Law | Issue-spotting,<br/> Rule-recall,<br/> Rule-application,<br/> Rule-conclusion,<br/> Interpretation,<br/> Rhetorical-understanding | LegalBench | Acc, EM | GPT-4, GPT-3.5, <br/> Claude-1, Incite, OPT<br/> Falcon, LLaMA-2, FLAN-T5...|
| \cite{fei2023lawbench} | Law | Legal QA, NER, <br/> Sentiment Analysis,<br/> Reading Comprehension | LawBench | F1, Acc,<br/> ROUGE-L,<br/> Normalized log-distance... | GPT-4,<br/> ChatGPT, <br/> InternLM-Chat,<br/> StableBeluga2...|
