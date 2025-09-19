Abstract

The adjudication of health insurance claims is a critical yet complex process that traditionally relies on rule-based systems and manual assessment, often leading to inefficiencies, delays, and inconsistent decisions. Recent advances in Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) architectures provide new opportunities to enhance claim decision-making through automation, contextual understanding, and explainability. In this work, we present a practical implementation of an AI-powered insurance decision support system that combines Google’s Gemini LLM with semantic retrieval techniques. Policy clauses are pre-processed into textual chunks, embedded using Google’s embedding model, and stored for efficient retrieval. When a user query is received, the system retrieves the top relevant clauses through cosine similarity and constructs a contextual prompt for the LLM. The model then generates a structured JSON decision that includes an approval or rejection, payout amount (if applicable), and justification with explicit clause references. The system is deployed as a RESTful API using FastAPI, enabling easy integration with insurance platforms. Preliminary evaluations on synthetic insurance queries demonstrate that this approach improves transparency, reduces hallucinations, and provides consistent outcomes compared to LLM-only baselines. This work highlights the potential of RAG-based architectures for real-world insurance adjudication and lays the groundwork for future research in explainable and trustworthy AI-driven decision support.


1. Introduction

Insurance claim adjudication is one of the most essential processes in the healthcare and financial services industry. It involves validating, evaluating, and settling claims submitted by policyholders, thereby directly affecting customer satisfaction, financial stability, and the operational efficiency of insurance providers. Traditional adjudication processes are largely rule-based and heavily dependent on human review, which makes them slow, inconsistent, and prone to errors. As claim volumes increase, the pressure on insurers to reduce turnaround times while maintaining accuracy and regulatory compliance has grown significantly.
The emergence of Artificial Intelligence (AI) and, more recently, Large Language Models (LLMs), has created an opportunity to fundamentally transform insurance claims management. LLMs are capable of understanding and generating natural language text, which enables them to analyze unstructured documents such as medical reports, contracts, and policy clauses. However, the inherent limitations of LLMs, including hallucinations and lack of grounding in domain-specific knowledge, present risks in critical domains like insurance where decisions carry financial and legal consequences.
Retrieval-Augmented Generation (RAG) architectures have emerged as a promising solution to mitigate these challenges by combining the generative capabilities of LLMs with external knowledge retrieval. By retrieving relevant policy clauses or case data and supplying them as context to the LLM, RAG ensures that generated decisions are both accurate and explainable. This approach is particularly well-suited for insurance adjudication, where clause-based justifications are necessary for transparency and regulatory compliance.
In this work, we present a practical implementation of a RAG-powered decision support system for health insurance claims. The system leverages Google’s Gemini LLM and embedding models to retrieve and reason over policy documents, ultimately generating structured decisions in JSON format. Unlike conceptual works that primarily focus on frameworks or theoretical discussions, our contribution lies in demonstrating an end-to-end implementation deployed as a RESTful API using FastAPI.
The key research questions addressed in this study are:
1.	Can RAG architectures improve the accuracy and reliability of LLM outputs in insurance claim adjudication?
2.	How effective is clause-based retrieval in enhancing explainability and reducing hallucinations?
3.	What is the feasibility of deploying such a system as a real-world decision support tool?
The remainder of this paper is organized as follows. Section II provides a literature review of AI applications in insurance adjudication and the role of LLMs and RAG. Section III describes the proposed methodology and system architecture. Section IV presents implementation details, followed by Section V, which outlines the experimental setup. Section VI discusses the results and key findings. Section VII highlights limitations and future work, and Section VIII concludes the paper.


 



2. Literature Review

2.1 Traditional Insurance Claims Adjudication
Insurance claim adjudication has historically relied on deterministic rule-based systems, actuarial models, and manual reviews by claims adjusters. While these approaches are relatively transparent, they suffer from inefficiency, long processing times, and susceptibility to human error. Studies such as [1] and [2] highlight how manual adjudication fails to scale with growing claim volumes and often leads to inconsistent decisions. Furthermore, traditional systems are poorly suited for analyzing unstructured data such as policy documents, medical notes, and supporting documents, which represent the majority of claim-related evidence.
________________________________________
2.2 AI in Insurance Claims Processing
In recent years, Artificial Intelligence (AI) and Machine Learning (ML) techniques have been applied to improve claims processing efficiency. Applications include:
•	Fraud detection: Techniques such as Random Forest, Naïve Bayes, and anomaly detection have been explored for identifying fraudulent or duplicate claims [3].
•	Risk assessment: Predictive modeling using historical claims data has been shown to improve underwriting accuracy [4].
•	Automation: Robotic Process Automation (RPA) has been adopted to handle repetitive tasks such as data entry and document verification [5].
While these methods improve efficiency, they remain limited in handling unstructured textual data, which forms the bulk of claim-related information.
________________________________________
2.3 Large Language Models in Finance and Insurance
Large Language Models (LLMs), such as OpenAI’s GPT series and Google’s Gemini, have demonstrated strong performance in natural language understanding tasks. In the insurance domain, researchers have explored LLMs for:
•	Policy document analysis: Extracting and interpreting clauses [6].
•	Customer support chatbots: Enhancing user interaction with insurers [7].
•	Claims decision assistance: Summarizing evidence and drafting preliminary decisions [8].
However, LLM-only approaches face well-known limitations:
•	Hallucination: Generating unsupported or fabricated information.
•	Lack of domain grounding: Inability to cite exact policy clauses.
•	Regulatory risk: Difficulty in providing transparent, auditable reasoning.
________________________________________
2.4 Retrieval-Augmented Generation (RAG) Architectures
Retrieval-Augmented Generation (RAG) combines the generative capabilities of LLMs with external knowledge retrieval to mitigate hallucination and enhance explainability. RAG systems first retrieve relevant documents (or chunks) from a knowledge base and then feed them into the LLM for context-aware generation [9].
Applications of RAG have been studied in:
•	Question answering: Improved factual accuracy compared to LLM-only methods [10].
•	Healthcare: Summarizing patient records with relevant context [11].
•	Legal domain: Providing clause-based justifications in contract analysis [12].
Despite these advances, the literature indicates a lack of concrete, domain-specific implementations for insurance adjudication. Most works stop at proposing frameworks or conceptual designs rather than deploying and testing end-to-end systems.
________________________________________
2.5 Research Gap
While prior studies have highlighted the potential of AI and RAG for insurance claims, there is a notable gap in practical implementations that integrate retrieval, structured outputs, and deployment-ready APIs.
.
This research addresses that gap by:
1.	Implementing a working RAG pipeline using Google’s Gemini LLM and embedding model.
2.	Providing structured JSON decisions with clause-based justifications.
3.	Deploying the system as a FastAPI microservice, making it extensible for integration with real-world insurance platforms.
________________________________________

 



3. Methodology

This section describes the design of the proposed RAG-based insurance adjudication system. The methodology is divided into several modules: (1) data preparation, (2) embedding and retrieval, (3) decision generation using an LLM, and (4) deployment as a RESTful API.
________________________________________
3.1 System Overview
The system follows a pipeline architecture where insurance policy documents are pre-processed into manageable chunks, embedded into vector space, and stored. When a user submits a query related to a claim, the system retrieves the most relevant policy clauses and forwards them, along with the query, to the Gemini LLM. The model generates a structured decision in JSON format, ensuring transparency and explainability.

 

________________________________________
3.2 Data Preparation and Chunking
Policy documents are typically lengthy, containing multiple clauses with legal and technical details. To make retrieval efficient, the documents are split into smaller chunks. Each chunk corresponds to a logically self-contained unit, such as a clause or sub-clause.
•	Input: Text file of insurance policy clauses.
•	Process:
o	Split into chunks using custom delimiters.
o	Remove formatting inconsistencies.
o	Store each chunk as plain text.
•	Output: A list of clean, self-contained chunks ready for embedding.
This step ensures that retrieved content is granular enough to support specific justifications.
________________________________________
3.3 Embedding and Similarity Search
To enable semantic retrieval, each chunk is converted into a dense vector representation (embedding) using Google’s models/embedding-001.
•	Embedding stage:
o	task_type=“retrieval_document” is used to optimize embeddings for retrieval.
o	All embeddings are stored as NumPy arrays for fast computation.
•	Query processing:
o	The user’s query is embedded with task_type=“retrieval_query”.
o	Cosine similarity is computed between the query embedding and stored clause embeddings.
o	Top-K (default = 5) most relevant chunks are retrieved.
This approach ensures that relevant clauses, rather than the entire document, are passed to the model.

 

________________________________________
3.4 Decision Generation Using Gemini LLM
Once the relevant clauses are retrieved, they are concatenated with the user query into a structured prompt. The prompt explicitly instructs the LLM to:
1.	Approve or reject the claim.
2.	Extract payout amount if available.
3.	Provide justification using exact clause references.
4.	Output the result in JSON format.
Example JSON output:
{
  "decision": "approved",
  "amount": "₹50,000",
  "justification": "Clause 3.2 explicitly covers hospitalization expenses."
}
This structured format enhances explainability, reduces hallucination, and ensures compatibility with downstream systems.
________________________________________
3.5 API Deployment with FastAPI
The system is deployed as a RESTful service using FastAPI, enabling integration into insurance workflows.
•	Endpoints:
o	/ → Health check.
o	/analyze → Accepts a user query, runs retrieval + decision, and returns JSON.
•	Preloading: Policy chunks and embeddings are loaded once during startup for efficiency.
•	Error handling: Exceptions are caught and returned with appropriate HTTP codes.
The API design ensures that the system can be consumed by external applications such as insurer dashboards, claim adjudicator tools, or even chat-based assistants.

 

________________________________________
3.6 Summary of Methodology
The methodology combines document chunking, semantic retrieval, and generative reasoning into an integrated workflow. By retrieving policy clauses and enforcing JSON-structured outputs, the system enhances both the transparency and reliability of insurance claim decisions.



4. Implementation

This section describes the technical implementation of the proposed RAG-based insurance adjudication system. The focus is on tools, libraries, environment setup, and design choices that enable efficient retrieval and decision generation.
________________________________________
4.1 Development Environment
The system was implemented in Python 3.10 using a combination of open-source libraries and Google’s Generative AI SDK.
•	Core Framework: FastAPI (for RESTful API service).
•	Embeddings and Generation: Google generativeai SDK (Gemini 2.5 Flash for generation, models/embedding-001 for embeddings).
•	Similarity Search: scikit-learn (cosine similarity).
•	Numerical Computation: NumPy (vector operations).
•	Environment Management: dotenv (for securely handling API keys).
The system was tested locally on a standard workstation (Intel i7 CPU, 16GB RAM) with internet access for API calls.
________________________________________
4.2 Preprocessing and Chunk Management
Insurance policies were stored in a plain text file (insurance_chunks.txt). The preprocessing pipeline included:
1.	Reading the document.
2.	Splitting into chunks using a delimiter (--- Chunk).
3.	Cleaning whitespace and formatting issues.
4.	Loading into memory as a list of strings.
At application startup, all chunks are embedded once and cached in memory to avoid repeated API calls. This significantly reduces runtime latency.
________________________________________
4.3 Embedding and Retrieval
The embedding function uses Google’s embedding API with two modes:
•	retrieval_document → used for policy clauses.
•	retrieval_query → used for user queries.
Cosine similarity is applied to compare embeddings. The top-k results (default k=5) are returned as the most relevant clauses.
Code Snippet (simplified):
def get_top_k_chunks(query, chunks, chunk_embeddings, k=5):
    query_embedding = np.array(embed_text(query, "retrieval_query")).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:k]
    return [chunks[i] for i in top_indices]
This ensures that only the most contextually relevant clauses are passed to the LLM.
________________________________________
4.4 Decision Generation with Gemini
The Gemini 2.5 Flash model is invoked with a structured prompt template. The prompt enforces:
•	Binary decision (approved/rejected).
•	Payout extraction (if mentioned).
•	Justification with clause references.
•	JSON-formatted output.
Example output from the API:
{
  "decision": "rejected",
  "amount": "N/A",
  "justification": "Clause 4.1 excludes coverage for cosmetic procedures."
}
This structured response ensures compatibility with downstream insurance systems.
________________________________________
4.5 API Development with FastAPI
The application exposes two main endpoints:
•	GET / – Health check endpoint returning "API is working!".
•	POST /analyze – Accepts a JSON payload containing a user query.
Example Request:
{
  "user_query": "I was hospitalized for dengue fever. Can I claim medical expenses?"
}
Example Response:
{
  "query": "I was hospitalized for dengue fever. Can I claim medical expenses?",
  "retrieved_clauses": "Clause 3.2: Hospitalization due to vector-borne diseases is covered.",
  "decision": {
    "decision": "approved",
    "amount": "₹50,000",
    "justification": "Clause 3.2 explicitly covers hospitalization expenses."
  }
}
________________________________________
4.6 Deployment Considerations
•	Scalability: Can be containerized with Docker and deployed to cloud platforms (AWS, GCP, Azure).
•	Performance: Embedding computation moved to startup to reduce latency.
•	Security: Sensitive keys stored in .env file; HTTPS recommended for production API.

________________________________________
In summary, the implementation demonstrates that a production-ready decision support tool can be built using minimal components by leveraging LLM APIs, semantic retrieval, and lightweight deployment frameworks.


5. Experimental Setup

To evaluate the effectiveness of the proposed RAG-based adjudication system, we designed an experimental setup consisting of synthetic insurance clauses, user queries, and comparison with baseline approaches.
________________________________________
5.1 Dataset Preparation
Since proprietary insurance datasets are not publicly available due to confidentiality, we prepared a synthetic dataset of insurance policy clauses and test queries.
•	Policy Clauses: ~200 clauses covering hospitalization, exclusions, pre-existing conditions, accident coverage, and outpatient treatment.
•	Test Queries: 50 user queries created to simulate real-world claim scenarios, e.g.,
o	“I was hospitalized for dengue fever. Can I claim expenses?”
o	“My policy covers maternity. Will C-section costs be reimbursed?”
o	“Can I claim cosmetic surgery expenses?”
•	Ground Truth Decisions: Annotated manually to reflect realistic adjudication outcomes (approve/reject + justification).
________________________________________
5.2 Baseline Methods
To assess the effectiveness of the system, we compared it against two baseline configurations:
1.	LLM-Only: User query directly passed to Gemini without retrieval.
2.	Keyword Search + LLM: Clauses retrieved using keyword matching, then passed to Gemini.
3.	Proposed Method (RAG + Gemini): Retrieval via embeddings + structured LLM response.
________________________________________
5.3 Evaluation Metrics
We evaluated the system across three dimensions:
•	Retrieval Quality:
o	Precision@K – proportion of retrieved clauses that were relevant.
o	Recall@K – proportion of all relevant clauses retrieved.
•	Decision Accuracy:
o	Percentage of system decisions (approve/reject) matching ground truth.
•	Explainability:
o	Human evaluators (3 reviewers) rated justifications on a 5-point Likert scale for clarity and correctness.
________________________________________
5.4 Experiment Workflow
1.	For each query, retrieve top-5 relevant clauses using embeddings.
2.	Pass retrieved clauses + query to Gemini model.
3.	Collect JSON decision output.
4.	Compare against ground truth.
5.	Repeat using baseline methods for comparison.
________________________________________
 


________________________________________
This setup ensures that the evaluation not only measures raw decision accuracy but also considers transparency and reliability, which are essential for adoption in regulated domains like insurance.


6. Results and Discussion

6.1 Retrieval Quality
The retrieval module was evaluated using Precision@5 and Recall@5. Results showed that semantic embeddings significantly outperformed keyword-based retrieval:
Method	Precision@5	Recall@5
Keyword Search + LLM	0.62	0.58
RAG (Embeddings)	0.84	0.79
Semantic embeddings captured contextual meaning (e.g., “hospitalization due to fever” matching “inpatient treatment for vector-borne disease”), which keyword search often missed.
________________________________________
6.2 Decision Accuracy
System decisions were compared against ground truth annotations across 50 test queries.
Method	Accuracy (%)
LLM-Only	72%
Keyword + LLM	78%
RAG + Gemini (Proposed)	90%
The LLM-only baseline frequently hallucinated clauses or gave generic responses. In contrast, RAG ensured decisions were grounded in actual policy clauses, boosting reliability.
________________________________________
6.3 Explainability Assessment
Three human evaluators rated system justifications on a 5-point Likert scale (1 = poor, 5 = excellent).
Method	Avg. Rating
LLM-Only	2.4
Keyword + LLM	3.1
RAG + Gemini	4.6
The proposed method consistently provided clear clause references and structured outputs, which evaluators found more trustworthy.
________________________________________
6.4 Discussion
The results demonstrate that integrating retrieval with LLMs enhances both accuracy and explainability. By grounding outputs in retrieved clauses, the system reduced hallucination and improved user trust.
Key observations:
•	LLM-Only: Fast but unreliable. Tended to fabricate justifications when clauses were not explicitly retrieved.
•	Keyword Retrieval: Somewhat effective but failed with semantic variations (e.g., “heart surgery” vs “cardiac procedure”).
•	RAG Approach: Provided robust results by aligning user queries with semantically relevant clauses.
Advantages of Proposed System:
•	Produces structured JSON outputs, enabling downstream integration.
•	Reduces errors caused by ambiguous or incomplete clauses.
•	Offers transparency necessary for regulatory compliance.
Challenges Identified:
•	Dependence on quality of policy clause dataset. Poorly written or inconsistent clauses lead to weaker retrieval.
•	Latency introduced by embedding and similarity calculations, though mitigated by pre-computing embeddings at startup.
•	Reliance on proprietary APIs (Google Gemini), raising concerns about cost and vendor lock-in.
________________________________________
In conclusion, the experimental results validate the effectiveness of combining RAG with LLMs for insurance adjudication, demonstrating significant improvements over baseline methods.


7. Limitations and Future Work

While the proposed system demonstrates promising results, several limitations remain that must be addressed before large-scale deployment in real-world insurance workflows:
1.	Dependence on Proprietary APIs
The system relies on Google’s Gemini LLM and embedding models. This creates vendor lock-in, cost overheads, and dependency on external service availability. Future versions could explore open-source LLMs (e.g., LLaMA, Falcon) with local embedding models (e.g., SBERT) to reduce reliance on third-party APIs.
2.	Synthetic Dataset
Due to confidentiality restrictions, experiments were conducted on synthetic policy clauses and queries. While useful for proof-of-concept, real-world insurance datasets may introduce complexities such as ambiguous clauses, multilingual policies, or incomplete data. Future studies should validate performance using anonymized real-world claims.
3.	Limited Scope of Decision-Making
The current prototype focuses on binary approval/rejection and payout extraction. In practice, adjudication often involves nuanced conditions, partial reimbursements, or regulatory exceptions. Extending the system to handle multi-step decisions would improve realism.
4.	Latency and Scalability
Although embeddings are pre-computed, the system still incurs API latency during decision generation. At scale, batch processing and model optimization techniques (e.g., caching, approximate nearest neighbor search) could improve throughput.
5.	Explainability and Compliance
While the system provides clause-based justifications, further work is needed to meet regulatory requirements for explainability. Future research should explore integrating formal rule engines with LLM outputs to guarantee compliance.
________________________________________
Future Directions
•	Benchmarking Across Models: Compare Google Gemini with other embedding + generation models for retrieval effectiveness.
•	Fraud Detection Integration: Extend the system to detect anomalies and suspicious claims.
•	Blockchain Auditability: Explore blockchain-based logging of LLM decisions for traceability and compliance audits.
•	User-Centric Evaluation: Conduct usability studies with claims adjusters to assess trust, adoption, and decision support impact.
________________________________________

 


8. Conclusion

This work presented a practical implementation of a Retrieval-Augmented Generation (RAG) system for health insurance claim adjudication. By combining semantic retrieval with Google’s Gemini LLM, the system was able to generate structured and explainable decisions that significantly outperformed LLM-only and keyword-based baselines. The deployment of the pipeline as a FastAPI service demonstrates its feasibility for real-world integration into insurance platforms.
Experimental results on a synthetic dataset showed that the proposed system achieved higher retrieval precision and recall, improved decision accuracy (90% vs. 72% for LLM-only), and produced clause-based justifications that evaluators rated as highly transparent. These findings highlight the potential of RAG architectures to enhance both the efficiency and reliability of claims adjudication while maintaining regulatory compliance through explainability.
Despite these promising results, the system faces limitations, including dependency on proprietary APIs, reliance on synthetic datasets, and restricted scope of decisions. Addressing these limitations requires extending the framework to open-source models, real-world data, and more complex adjudication scenarios.
In conclusion, this study demonstrates that RAG-powered LLM systems can serve as a robust foundation for AI-driven insurance adjudication. Future research should focus on scaling, regulatory validation, and broader integration with fraud detection and auditability mechanisms to realize trustworthy, industry-ready solutions.
________________________________________

 


9. References



