# RAGFlow Data Model - Sequence Diagrams

## 1. User Registration & Tenant Creation

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant API as API App
    participant US as UserService
    participant DB as Database

    C->>API: POST /register {email, password}
    API->>US: create_user()
    US->>DB: INSERT User (id=uuid)
    DB-->>US: OK
    US->>DB: INSERT Tenant (id=user.id)
    DB-->>US: OK
    US->>DB: INSERT UserTenant (user_id, tenant_id, role=OWNER)
    DB-->>US: OK
    US-->>API: user_id
    API-->>C: {user_id, access_token}

    Note over DB: User.id = Tenant.id (1:1 mapping)
```

## 2. Knowledge Base Creation

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant KB as kb_app
    participant KBS as KBService
    participant DB as Database
    participant VDB as VectorDB

    C->>KB: POST /kb/create {name, parser_id} + JWT

    Note over KB: Extract tenant_id from current_user.id

    KB->>KBS: create_with_name(name, tenant_id)
    KBS->>DB: INSERT Knowledgebase (tenant_id, created_by)
    DB-->>KBS: OK

    KBS->>VDB: createIdx("ragflow_{tenant_id}")
    VDB-->>KBS: OK

    KBS-->>KB: kb_id
    KB-->>C: {kb_id}
```

## 3. Document Upload & Processing

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant DOC as doc_app
    participant DS as DocService
    participant TS as TaskService
    participant DB as Database
    participant S3 as MinIO
    participant VDB as VectorDB

    C->>DOC: POST /document/upload {kb_id, file}
    DOC->>S3: save_file()
    S3-->>DOC: file_location

    DOC->>DS: create_document()
    DS->>DB: INSERT Document (kb_id, created_by, status=pending)
    DB-->>DS: doc_id

    DS->>TS: queue_task(doc_id)
    TS->>DB: INSERT Task (doc_id, priority)
    DB-->>TS: task_id

    DOC-->>C: {doc_id}

    rect rgb(240, 240, 240)
        Note over TS,VDB: ASYNC TASK EXECUTOR

        TS->>DB: poll_task()
        DB-->>TS: task

        TS->>S3: get_file()
        S3-->>TS: file_content

        Note over TS: deepdoc.parse() - PDF to Text
        Note over TS: chunk_text() - Split into chunks
        Note over TS: embed_chunks() - Generate vectors

        TS->>VDB: index_chunks(kb_id, doc_id, chunks[])
        VDB-->>TS: OK

        TS->>DB: UPDATE Document (status=done, chunk_num=N)
        DB-->>TS: OK
    end
```

## 4. Search / Retrieval

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant SA as search_app
    participant D as Dealer
    participant EM as EmbedModel
    participant VDB as VectorDB

    C->>SA: POST /search {query, kb_ids}

    Note over SA: idx_name = ragflow_{tenant_id}

    SA->>D: search(query, kb_ids)
    D->>EM: encode_query(query)
    EM-->>D: query_vector

    D->>VDB: hybrid_search(filters: kb_id IN kb_ids)

    Note over VDB: 1. Vector similarity
    Note over VDB: 2. Keyword match
    Note over VDB: 3. Score fusion

    VDB-->>D: chunks[]
    D-->>SA: results[]
    SA-->>C: {chunks, scores}
```

## 5. RAG Chat End-to-End

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant DA as dialog_app
    participant DS as DialogService
    participant D as Dealer
    participant VDB as VectorDB
    participant LLM as LLM

    C->>DA: POST /chat {dialog_id, message}
    DA->>DS: get_dialog(dialog_id)
    DS-->>DA: dialog with kb_ids

    DA->>D: retrieval(query, kb_ids)
    D->>VDB: search()
    VDB-->>D: chunks[]
    D-->>DA: relevant_chunks[]

    Note over DA: build_prompt(chunks, history)

    DA->>LLM: chat_completion(prompt)

    loop Stream response
        LLM-->>DA: chunk
        DA-->>C: SSE chunk
    end

    DA->>DS: save_conversation()
    DS-->>DA: OK

    DA-->>C: SSE DONE
```

## 6. Multi-Tenant Data Isolation

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant Auth as Auth Middleware
    participant US as UserService
    participant DB as Database
    participant VDB as VectorDB
    participant S3 as FileStore

    C->>Auth: Request + Authorization JWT
    Auth->>Auth: jwt.loads() to access_token
    Auth->>US: query(access_token)
    US->>DB: SELECT User WHERE access_token=X
    DB-->>US: User
    US-->>Auth: current_user

    Note over Auth: current_user.id = tenant_id

    rect rgb(255, 245, 238)
        Note over DB,S3: TENANT ISOLATION POINTS

        Auth->>DB: WHERE tenant_id = X
        Auth->>VDB: Index ragflow_X and Filter kb_id IN list
        Auth->>S3: Path /{tenant_id}/*
    end
```

## 7. Entity Relationship Diagram

```mermaid
erDiagram
    User ||--o{ UserTenant : has
    Tenant ||--o{ UserTenant : has
    User ||--o{ Knowledgebase : creates
    Tenant ||--o{ Knowledgebase : owns
    Knowledgebase ||--o{ Document : contains
    User ||--o{ Document : uploads
    Document ||--o{ Chunk : splits_into
    Knowledgebase ||--o{ Dialog : linked_to
    Tenant ||--o{ Dialog : owns

    User {
        string id PK
        string email
        string access_token
        string password
        string status
    }

    Tenant {
        string id PK
        string name
        string llm_id
        string embd_id
        string status
    }

    UserTenant {
        string id PK
        string user_id FK
        string tenant_id FK
        string role
        string invited_by
    }

    Knowledgebase {
        string id PK
        string tenant_id FK
        string created_by FK
        string name
        string permission
        string embd_id
        string parser_id
        int doc_num
        int chunk_num
    }

    Document {
        string id PK
        string kb_id FK
        string created_by FK
        string name
        string location
        string source_type
        string status
        int chunk_num
    }

    Chunk {
        string id PK
        string kb_id FK
        string doc_id FK
        text content
        blob embedding
        int position
        int page_num
    }

    Dialog {
        string id PK
        string tenant_id FK
        json kb_ids
        string name
        string llm_id
        json history
    }
```

## 8. Document Processing Pipeline

```mermaid
flowchart TD
    subgraph Upload["Upload Phase"]
        A[Client Upload] --> B[Save to MinIO]
        B --> C[Create Document Record]
        C --> D[Queue Task]
    end

    subgraph Process["Processing Phase"]
        D --> E[Task Executor Polls]
        E --> F[Load File from MinIO]
        F --> G[deepdoc Parse]

        subgraph Parsing["Document Parsing"]
            G --> G1[PDF Layout Analysis]
            G1 --> G2[OCR if needed]
            G2 --> G3[Table Extraction]
            G3 --> G4[Text Extraction]
        end

        G4 --> H[Chunking]

        subgraph Chunking["Text Chunking"]
            H --> H1[Token Split]
            H1 --> H2[Overlap Handling]
            H2 --> H3[Position Tracking]
        end
    end

    subgraph Index["Indexing Phase"]
        H3 --> I[Generate Embeddings]
        I --> J[Index to VectorDB]
        J --> K[Update Document Status]
    end

    subgraph Storage["Storage Layer"]
        B -.-> S3[(MinIO/S3)]
        C -.-> DB[(MySQL/PostgreSQL)]
        J -.-> VDB[(Elasticsearch)]
    end
```

## 9. Multi-KB Federated Search

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant API as API
    participant D as Dealer
    participant VDB as VectorDB

    C->>API: POST /search {query, kb_ids: [kb1, kb2, kb3]}

    API->>D: search(query, [kb1, kb2, kb3])

    Note over D: Single index per tenant ragflow_{tenant_id}

    D->>VDB: hybrid_search index ragflow_{tenant_id} filter kb_id IN [kb1, kb2, kb3]

    VDB->>VDB: Vector similarity search
    VDB->>VDB: Keyword BM25 search
    VDB->>VDB: Score fusion (vector * 0.7 + keyword * 0.3)

    VDB-->>D: merged_results[]

    Note over D: Results contain chunks from all 3 KBs

    D-->>API: results[]
    API-->>C: {chunks from kb1, kb2, kb3}
```

## 10. Permission Check Flow

```mermaid
flowchart TD
    A[Request to Access KB] --> B{Check current_user}

    B -->|Authenticated| C{KB.created_by == user.id?}
    B -->|Not Auth| X[401 Unauthorized]

    C -->|Yes| Y[Full Access - Owner]
    C -->|No| D{KB.permission?}

    D -->|me| X2[403 Forbidden]
    D -->|team| E{User in same tenant?}

    E -->|Yes| F{Check UserTenant.role}
    E -->|No| X3[403 Forbidden]

    F -->|OWNER/ADMIN| Y2[Full Access]
    F -->|NORMAL| Y3[Read Access]
    F -->|INVITE| Y4[Limited Access]
```

## 11. GraphRAG Processing Flow

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant API as API
    participant GR as GraphRAG
    participant LLM as LLM
    participant VDB as VectorDB

    C->>API: POST /kb/graphrag {kb_id}
    API->>GR: run_graphrag_for_kb(kb_id)

    loop For each document
        GR->>VDB: get_chunks(doc_id)
        VDB-->>GR: chunks[]

        GR->>LLM: extract_entities(chunks)
        LLM-->>GR: entities[]

        GR->>LLM: extract_relationships(chunks)
        LLM-->>GR: relationships[]

        GR->>GR: build_subgraph()
    end

    GR->>GR: merge_subgraphs()
    GR->>GR: entity_resolution()
    GR->>GR: community_detection()

    GR->>VDB: index_graph(entities, relations, communities)
    VDB-->>GR: OK

    GR-->>API: {graph_task_id}
    API-->>C: {status: done}
```

## 12. Complete Data Flow Overview

```mermaid
flowchart TB
    subgraph Client["Client Layer"]
        Web[Web UI]
        SDK[Python SDK]
        REST[REST API]
    end

    subgraph API["API Layer"]
        Auth[Authentication]
        KB[KB App]
        Doc[Document App]
        Chat[Dialog App]
        Search[Search App]
    end

    subgraph Service["Service Layer"]
        KBS[KnowledgebaseService]
        DS[DocumentService]
        DLS[DialogService]
        TS[TaskService]
    end

    subgraph Core["Core Processing"]
        DP[deepdoc Parser]
        EMB[Embedding Model]
        LLM[LLM Chat]
        GR[GraphRAG]
    end

    subgraph Storage["Storage Layer"]
        DB[(MySQL/PostgreSQL)]
        VDB[(Elasticsearch/Infinity)]
        S3[(MinIO/S3)]
        Redis[(Redis)]
    end

    Web --> Auth
    SDK --> Auth
    REST --> Auth

    Auth --> KB
    Auth --> Doc
    Auth --> Chat
    Auth --> Search

    KB --> KBS
    Doc --> DS
    Chat --> DLS
    Search --> DLS

    KBS --> DB
    DS --> DB
    DS --> TS
    DLS --> DB

    TS --> DP
    DP --> EMB
    EMB --> VDB

    DLS --> LLM
    DLS --> VDB

    DS --> S3
    TS --> Redis
    GR --> VDB
```
