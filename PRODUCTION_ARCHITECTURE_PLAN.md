# PRODUCTION ARCHITECTURE PLAN
## Scaling the Cephalometric AI System from MVP to Enterprise

**Version**: 1.0  
**Last Updated**: 2026-05-11  
**Status**: Planning Phase (Awaiting Clinical Annotations)

---

## Overview

This document outlines a phased approach to evolve the current Singdent Cephalometric AI system from a streamlit-based Minimum Viable Product (MVP) to a production-grade microservices architecture. The transformation is designed to occur while awaiting clinical annotations for model training, leveraging this period to build robust enterprise infrastructure.

The current system consists of:
- **Backend Logic**: `src/phase3/biomechanics.py` (mathematical engine for cephalometric analysis)
- **Frontend**: `app.py` (streamlit-based dashboard for demonstration)
- **Tight Coupling**: Direct function calls between UI and business logic

The target architecture will decouple these concerns, enabling independent scaling, deployment, and integration with clinical workflows.

---

## Phase 1: FastAPI Backend (AI & Logic API)

### Objective
Decouple `biomechanics.py` and the Zhang 2021 logic from streamlit into a standalone API server that can serve multiple clients (web, mobile, clinical systems) while maintaining the existing analytical capabilities.

### Technical Stack
- **Framework**: FastAPI (modern, fast, async-first python web framework)
- **ASGI Server**: Uvicorn (lightning-fast ASGI server implementation)
- **Data Validation**: Pydantic (for request/response modeling)
- **Documentation**: Automatic OpenAPI/Swagger generation

### Implementation Plan

#### 1. Project Structure
```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # fastapi application entrypoint
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       └── endpoints.py    # api route definitions
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py           # configuration management
│   │   └── dependencies.py     # dependency injection
│   ├── services/
│   │   ├── __init__.py
│   │   └── analysis_service.py # wrapper for biomechanics logic
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py          # pydantic models for requests/responses
│   └── utils/
│       ├── __init__.py
│       └── image_processing.py # image handling utilities
├── requirements.txt
├── dockerfile
└── readme.md
```

#### 2. Core API Endpoints

**POST `/api/v1/analyze`**
- **Purpose**: perform cephalometric analysis on uploaded image
- **Request Body**:
  ```json
  {
    "image": "base64_encoded_string_or_file_upload",
    "mm_per_pixel": 0.0984,
    "bone_analysis_side": "both"  // "left", "right", or "both"
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "data": {
      "metrics": {
        "u1_pp_angle_deg": 112.5,
        "lb_apex_dist_mm": 1.2,
        "pb_apex_dist_mm": 0.9
      },
      "classification": {
        "root_apix_position": "midway",
        "incisor_condition": "normal inclination with centered apex",
        "preferred_biomechanics": "bodily movement (translation)",
        "biomechanics_to_avoid": "uncontrolled tipping",
        "clinical_implication": "most favorable condition"
      },
      "bone_thickness": {
        "labial_min_mm": 0.4,
        "labial_mean_mm": 0.6,
        "palatal_min_mm": 0.3,
        "palatal_mean_mm": 0.5,
        "status": "adequate"
      }
    },
    "processing_info": {
      "image_dimensions": [1729, 2048],
      "landmarks_detected": 10,
      "processing_time_ms": 45
    }
  }
  ```

**GET `/api/v1/health`**
- **Purpose**: health check endpoint for load balancers and monitoring
- **Response**:
  ```json
  {
    "status": "healthy",
    "timestamp": "2026-05-11t10:30:00z",
    "version": "1.0.0"
  }
  ```

**GET `/api/v1/models/info`**
- **Purpose**: metadata about available ai models (for future integration)
- **Response**:
  ```json
  {
    "landmark_detection": {
      "model": "hrnet-w32",
      "status": "not_loaded",  // will load when annotations available
      "input_size": [256, 256]
    },
    "segmentation": {
      "model": "unet-resnet34",
      "status": "not_loaded",
      "num_classes": 3
    }
  }
  ```

#### 3. Service Layer Design

The `analysis_service` will act as a thin wrapper around the existing biomechanics module:

```python
# app/services/analysis_service.py
class analysis_service:
    def __init__(self):
        # import existing biomechanics functions - no modifications needed
        from src.phase3.biomechanics import (
            calculate_metrics,
            classify_treatment,
            bonethicknesscalculator
        )
        self.calculate_metrics = calculate_metrics
        self.classify_treatment = classify_treatment
        self.bonethicknesscalculator = bonethicknesscalculator
    
    def analyze_image(self, image_array: np.ndarray, mm_per_pixel: float) -> dict:
        """
        main analysis pipeline that coordinates the existing biomechanics functions.
        
        args:
            image_array: numpy array of the cephalometric image
            mm_per_pixel: calibration factor for the image
            
        returns:
            dictionary containing all analysis results
        """
        # step 1: extract landmarks (placeholder - will integrate with trained model)
        landmarks = self._extract_landmarks(image_array)
        
        # step 2: calculate metrics using existing function
        metrics = self.calculate_metrics(landmarks, mm_per_pixel)
        
        # step 3: get classification using existing function
        classification = self.classify_treatment(
            metrics["u1_pp_angle_deg"],
            metrics["lb_apex_dist_mm"],
            metrics["pb_apex_dist_mm"]
        )
        
        # step 4: calculate bone thickness using existing class
        bone_calc = self.bonethicknesscalculator(landmarks)
        bone_thickness = bone_calc.calculate_all_thicknesses()
        
        return {
            "metrics": metrics,
            "classification": classification,
            "bone_thickness": bone_thickness
        }
```

#### 4. Key Design Decisions

1. **zero changes to core logic**: the existing `biomechanics.py` module requires no modifications. it will be imported and used as-is.
2. **async-first design**: fastapi's async capabilities will be leveraged for concurrent request handling, though the cpu-intensive analysis will run in thread pools to avoid blocking.
3. **validation layer**: pydantic models will ensure data integrity at api boundaries.
4. **extensible architecture**: designed to easily integrate trained ai models for landmark detection when annotations become available.

#### 5. Immediate Next Steps (Post-Planning)
1. create backend repository structure
2. implement basic fastapi application with health check
3. add the analysis endpoint using mocked landmark data
4. generate automatic api documentation via swagger ui
5. create dockerfile for containerization

---

## Phase 2: Astro Frontend (Clinical Dashboard)

### Objective
Replace the rigid streamlit ui with a fast, fully customizable web dashboard built with modern web technologies that provides superior performance, flexibility, and user experience.

### Technical Stack
- **Framework**: astro 3.0+ (island architecture for optimal performance)
- **UI Framework**: react (for interactive components) or svelte (alternative based on team preference)
- **Styling**: tailwind css (utility-first css for rapid ui development)
- **State Management**: zustand or react context (lightweight state management)
- **HTTP Client**: fetch api or axios for communicating with fastapi backend
- **Build Tools**: built-in astro optimizer (no additional bundler needed)

### Implementation Plan

#### 1. Project Structure
```
frontend/
├── public/
│   └── favicon.svg
├── src/
│   ├── components/
│   │   ├── layout/
│   │   │   ├── header.astro
│   │   │   └── footer.astro
│   │   ├── ui/
│   │   │   ├── uploadzone.jsx
│   │   │   ├── imageviewer.jsx
│   │   │   ├── metriccard.jsx
│   │   │   └── classificationbadge.jsx
│   │   └── pages/
│   │       └── index.astro     # main dashboard page
│   ├── lib/
│   │   ├── api.js              # api client wrapper
│   │   ├── utils.js            # image processing utilities
│   │   └── constants.js        # app-wide constants
│   ├── styles/
│   │   └── global.css          # tailwind base styles
│   └── content/
│       └── config.ts           # content collection configuration
├── astro.config.mjs
├── package.json
├── tailwind.config.cjs
└── readme.md
```

#### 2. Core UI Components

**uploadzone component**
- drag-and-drop or click-to-upload interface
- file validation (image types, size limits)
- upload progress indication
- integration with `api.js` to send to `/api/v1/analyze`

**imageviewer component**
- displays uploaded cephalometric radiograph
- overlays detected landmarks (when available)
- shows measurement lines (u1 axis, palatal plane)
- interactive zoom/pan capabilities
- toggleable annotation layers

**metriccard component (reusable)**
- displays single metric with label, value, unit
- color-coded based on clinical thresholds:
  - green: normal/adequate
  - yellow: caution/monitor
  - red: abnormal/requires attention
- tooltip with detailed explanation on hover

**classificationbadge component**
- shows the treatment classification result
- color-coded based on risk level
- expandable to show full classification details
- links to educational resources when clicked

#### 3. Data Flow Architecture

```mermaid
sequencediagram
    participant user
    participant frontend as astro frontend
    participant backend as fastapi api
    participant db as database (future)
    
    user->>frontend: uploads cephalometric image
    frontend->>backend: post /api/v1/analyze (image + params)
    backend->>backend: process using biomechanics logic
    backend-->>frontend: json response with results
    frontend->>frontend: update ui components
    frontend->>user: display results with visualizations
```

#### 4. Key Features

1. **real-time validation**: client-side validation before sending to api
2. **optimistic ui**: immediate feedback while waiting for processing
3. **error handling**: graceful degradation and user-friendly error messages
4. **responsive design**: works on desktop tablets and mobile devices
5. **accessibility**: wcag 2.1 compliant (aria labels, keyboard navigation)
6. **performance**: 
   - zero-js by default (astro islands)
   - only loads interactive components when needed
   - optimized image loading and caching

#### 5. Integration with Backend

The frontend will communicate with the backend through a well-defined api client:

```javascript
// src/lib/api.js
class cephalometricapi {
    constructor(baseurl = '') {
        this.baseurl = baseurl;
    }
    
    async analyzeimage(formdata) {
        const response = await fetch(`${this.baseurl}/api/v1/analyze`, {
            method: 'post',
            body: formdata
        });
        
        if (!response.ok) {
            throw new error(`api error: ${response.status}`);
        }
        
        return await response.json();
    }
    
    async healthcheck() {
        const response = await fetch(`${this.baseurl}/api/v1/health`);
        return await response.json();
    }
}

// usage in components
const api = new cephalometricapi(import.meta.env.public_api_url);
const result = await api.analyzeimage(formdata);
```

#### 6. Design Principles

1. **clinical focus**: clean, professional interface suitable for medical environments
2. **information hierarchy**: most critical findings presented first
3. **minimal cognitive load**: progressive disclosure of complex information
4. **consistency**: unified design language across all components
5. **feedback**: clear visual feedback for all user interactions

#### 7. Immediate Next Steps (Post-Planning)
1. initialize astro project with typescript and tailwind
2. create basic layout and routing structure
3. implement uploadzone and imageviewer components
4. build api client wrapper
5. create dashboard layout with metric display area
6. add responsive breakpoints and mobile optimization

---

## Phase 3: Enterprise Deployment & Cvat Nuclio Integration

### Objective
Containerize the application for consistent deployment across environments and integrate ai services directly into the clinical workflow via cvat's serverless framework (nuclio).

### Technical Stack
- **Containerization**: docker (application packaging)
- **Orchestration**: docker compose (local/dev), kubernetes (production)
- **Service Mesh**: internal service communication
- **CVAT Integration**: nuclio serverless functions for ai inference
- **Monitoring**: prometheus/grafana (health and performance metrics)
- **Logging**: elk stack or similar centralized logging

### Implementation Plan

#### 1. Docker Compose Architecture

```yaml
# docker-compose.yml
version: '3.8'

services:
  # fastapi backend service
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - environment=development
      - log_level=info
    volumes:
      - ./backend:/app
      - model_data:/app/models
    restart: unless-stopped
    healthcheck:
      test: ["cmd", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # astro frontend service
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - public_api_url=http://backend:8000
    volumes:
      - ./frontend:/app
    restart: unless-stopped
    depends_on:
      backend:
        condition: service_healthy

  # optional: reverse proxy (traefik/nginx)
  # proxy:
  #   image: traefik:v2.10
  #   ports:
  #     - "80:80"
  #     - "8080:8080"  # dashboard
  #   volumes:
  #     - /var/run/docker.sock:/var/run/docker.sock
  #     - ./traefik.yml:/etc/traefik/traefik.yml
  #     - ./traefik:/etc/traefik

  # optional: database for storing historical analyses
  # db:
  #   image: postgres:15
  #   environment:
  #     - postgres_db=cephalometric
  #     - postgres_user=user
  #     - postgres_password=password
  #   volumes:
  #     - db_data:/var/lib/postgresql/data

volumes:
  model_data:
  db_data:
```

#### 2. Cvat Nuclio Integration Plan

Cvat (computer vision annotation tool) supports serverless functions through nuclio, allowing custom ai models to be integrated directly into the annotation workflow.

**integration strategy**:
1. package our ai models (when trained) as nuclio functions
2. register functions with cvat serverless framework
3. enable functions as "auto-annotation" tools in cvat ui
4. allow clinicians to trigger ai pre-labeling during manual annotation

**nuclio function structure**:
```
nuclio-function/
├── function.yaml          # nuclio function configuration
├── main.py                # entry point for nuclio
├── requirements.txt       # python dependencies
├── model/                 # trained model files (when available)
│   ├── landmark_detection/
  │   │   └── model.pth
  │   └── segmentation/
  │       └── model.pth
└── readme.md
```

**function.yaml**:
```yaml
name: cephalometric-analyzer
description: "cephalometric ai analysis service for cvat"
version: "0.1.0"
spec:
  runtime: python-3.9
  handler: main:handler
  events:
    - http:
        url: /analyze
        method: post
```

**main.py** (nuclio handler):
```python
import json
import nuclio_sdk
from cephalometric_service import analysis_service  # our packaged service

def handler(context, event):
    """nuclio entry point for http requests"""
    try:
        # parse incoming request
        body = json.loads(event.body)
        
        # initialize our analysis service
        analyzer = analysis_service()
        
        # process the request (adapt to nuclio's expected format)
        result = analyzer.analyze_from_cvat_format(body)
        
        # return response in format cvat expects
        return nuclio_sdk.response(
            body=json.dumps(result),
            headers={"content-type": "application/json"},
            status_code=200
        )
    except exception as e:
        context.logger.error_with(f"error processing request: {str(e)}")
        return nuclio_sdk.response(
            body=json.dumps({"error": str(e)}),
            headers={"content-type": "application/json"},
            status_code=500
        )
```

#### 3. Deployment Pipeline

```mermaid
flowchart td
    a[code commit] --> b{ci pipeline}
    b -->|tests pass| c[build docker images]
    c --> d[push to registry]
    d --> e[staging deployment]
    e -->|integration tests| f[production approval]
    f --> g[rolling update]
    g --> h[monitoring & alerts]
    
    subgraph development
        b
        c
        d
    end
    
    subgraph production
        e
        f
        g
        h
    end
```

#### 4. Key Infrastructure Components

**1. image optimization service**
- separate service for preprocessing images before analysis
- handles format conversion, resizing, normalization
- can be scaled independently based on workload

**2. results storage & retrieval**
- temporary storage for recent analyses (redis)
- long-term storage for audit trails (postgresql/s3)
- api endpoints for retrieving historical analyses

**3. monitoring & observability**
- health checks for all services
- latency and throughput metrics
- error rate tracking
- automated alerts for service degradation

**4. security considerations**
- api authentication (jwt/oauth2 for production)
- input validation and sanitization
- secure file upload handling
- audit logging for phi access (hipaa compliance preparation)
- regular security scanning of dependencies

#### 5. Cvat Integration Workflow

```mermaid
sequencediagram
    participant clinician
    participant cvat as cvat web interface
    participant nuclio as nuclio function
    participant storage as object storage
    
    clinician->>cvat: uploads cephalometric image
    cvat->>nuclio: triggers auto-annotation (cephalometric-analyzer)
    nuclio->>storage: loads pre-trained model (when available)
    nuclio->>storage: reads image from cvat storage
    nuclio->>nuclio: runs landmark detection + segmentation
    nuclio->>cvat: returns annotation data (points, polygons)
    cvat->>clinician: displays ai-generated annotations for review
    clinician->>cvat: edits/confirms annotations
    cvat->>storage: saves final annotated data
```

#### 6. Resource Requirements & Scaling

**Development Environment**:
- single docker compose file for local development
- minimal resource requirements (2gb ram, 2 cpu cores)
- hot-reload for rapid iteration

**Production Environment**:
- kubernetes deployment with resource requests/limits
- horizontal pod autoscaling based on cpu/utilization
- separate node pools for gpu-intensive tasks (when ai models active)
- backup and disaster recovery procedures

**estimated resource usage (per instance)**:
- backend (fastapi): 500mb ram, 0.5 cpu
- frontend (astro): 200mb ram, 0.2 cpu (mostly client-side)
- nuclio functions: scale to 0 when idle, burst to needed instances

#### 7. Immediate Next Steps (Post-Planning)
1. create dockerfiles for backend and frontend services
2. develop docker-compose.yml for local development
3. create basic nuclio function template structure
4. document environment variables and configuration options
5. outline ci/cd pipeline steps (github actions/gitlab ci)
6. create runbook for deployment and troubleshooting

---

## Cross-Phase Considerations

### Data Flow & Interfaces
All phases will communicate through well-defined, versioned apis:
- **internal**: backend services communicate via rest/grpc
- **external**: frontend communicates with backend via rest/json
- **cvat integration**: nuclio functions expose http endpoints for cvat

### Error Handling & Resilience
- **backend**: circuit breaker pattern for external dependencies
- **frontend**: retry mechanisms with exponential backoff
- **deployment**: health checks and automatic restart policies
- **data**: validation at every interface to prevent corruption

### Security & Compliance
- **data protection**: encryption at rest and in transit
- **access control**: role-based access control (rbac) for api
- **audit logging**: comprehensive logging for all phi access
- **regular updates**: automated dependency security scanning

### Performance Optimization
- **caching**: redis for frequently accessed data
- **image processing**: optimized libraries (opencv, pil/pillow)
- **concurrent processing**: async handling of multiple requests
- **static assets**: cdn distribution for frontend assets

---

## Success Criteria & Metrics

### Phase 1: FastAPI Backend
- [ ] api responds to health check within 100ms
- [ ] analysis endpoint returns results within 2 seconds (95th percentile)
- [ ] 99.9% uptime sla achieved in staging environment
- [ ] automatic api documentation generated and accurate
- [ ] stress testing shows handling of 10 concurrent requests

### Phase 2: Astro Frontend
- [ ] first contentful paint < 1.5s on 3g connection
- [ ] time to interactive < 3s on mid-tier mobile devices
- [ ] lighthouse performance score > 90
- [ ] accessibility score > 95 (wcag 2.1 aa)
- [ ] user satisfaction score > 4/5 in clinical testing

### Phase 3: Enterprise Deployment
- [ ] successful deployment to staging environment via docker-compose
- [ ] all services health-check passing after deployment
- [ ] nuclio function template successfully loads and responds
- [ ] rollback procedure tested and documented
- [ ] backup and recovery procedures validated

---

## Risk Assessment & Mitigation

| risk | probability | impact | mitigation strategy |
|------|-------------|--------|---------------------|
| delay in clinical annotations | high | medium | continue with mocked data; design for easy model swapping |
| performance bottlenecks in analysis | medium | high | profile early; optimize bottlenecks; consider gpu acceleration |
| integration complexity with cvat | medium | medium | start with simple nuclio function; iterate based on feedback |
| security vulnerabilities | low | high | implement security scanning; follow owasp guidelines; regular audits |
| team learning curve (new tech stack) | medium | low | allocate time for training; use well-documented technologies; pair programming |

---

## Timeline & Milestones

**note**: actual timing will depend on annotation availability and team capacity.

**month 1**:
- complete backend api implementation and testing
- develop initial frontend layout and components
- create docker-compose setup for local development

**month 2**:
- implement frontend-backend integration
- conduct usability testing with clinical stakeholders
- refine based on feedback

**month 3**:
- prepare cvat nuclio function templates
- finalize deployment documentation
- prepare for production rollout when annotations available

**ongoing**:
- weekly architecture reviews
- continuous integration and testing
- documentation updates
- knowledge sharing sessions

---

## Conclusion

This architecture plan provides a clear, phased approach to transforming the singdent cephalometric ai system from a demonstration mvp into a robust, enterprise-ready platform. By leveraging the existing well-designed biomechanics module as the core analytical engine, we can focus on building scalable, maintainable, and clinically useful infrastructure while awaiting the data needed to train our ai models.

The separation of concerns enabled by this architecture will:
1. allow independent scaling of analytical and presentation layers
2. enable integration with multiple client types (web, mobile, ehr)
3. facilitate continuous deployment and improvement
4. create a foundation for future ai-enhanced features
5. prepare the system for clinical validation and regulatory compliance

The modular approach ensures that each phase delivers immediate value while building toward the ultimate goal of a fully integrated clinical workflow solution.