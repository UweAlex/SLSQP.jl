# Strategic Paper 

## SLSQP.jl – Reproduction as Foundation, Product Line as Structure

### 1. Mission

Development of a pure Julia implementation of SLSQP that:

* is functionally and numerically **equivalent** to NLopt/SciPy-SLSQP.
* operates without C/Fortran wrappers.
* is robust and reproducible.
* integrates seamlessly into the existing Julia optimization ecosystem.

**SLSQP is the primary end product.**
Independent, stand-alone end products are intentionally created along the way.
Modernization occurs only after equivalence is secured.

---

### 2. Strategic Guiding Principles

#### 2.1 The Equivalence Axiom

Phases 0–3 serve exclusively for algorithmic reproduction.
Equivalence means:

* Comparable iteration paths.
* Comparable convergence rates.
* Comparable termination conditions.
* Comparable robustness with degenerate problems.

**NLopt/SciPy serve as the reference anchors.**
Any significant deviation is an investigation case—not a feature.

#### 2.2 Modern Implementation ≠ Modern Algorithmics

**Allowed:** Multiple Dispatch, type parameters, in-place APIs, clear workspace structures, idiomatic Julia design.
**Not allowed before Phase 4:** Alternative QP backends, different merit functions, altered damping strategies, algorithmic "improvements."

**The mathematics remains unchanged. Only the implementation is modern.**

#### 2.3 Purpose-Built Product Line

SLSQP is the goal. Intermediate products are intentionally crafted, independent tools.
**Potential Product Structure:**

1. **NNLS-Core**
2. **QP-Transformation-Toolkit**
3. **KKT-Diagnostics-Utilities**
4. **SLSQP-Integration**

Each intermediate product must be independently usable, have a clear API, be documented, and be maintainable. **No "waste-product" releases.**

#### 2.4 Ecosystem Integration, Not a Parallel World

SLSQP.jl shall be compatible with `Optimization.jl`, satisfy `SciMLBase` interfaces, utilize existing data structures, and not enforce proprietary problem formats.
**The goal is interchangeability, not secession.**

---

### 3. Phased Roadmap

#### Phase 0 – Reference Forensics

**Goal:** Secure exact algorithmic specification.
**Analysis of:** Original Kraft-Fortran, NLopt C-implementation, SciPy port.
**Documentation of:** Numerical constants, rank tolerances, termination criteria, damping factors, merit parameters, active-set logic.
**Output:** Technical Reference Document.

#### Phase 1 – NNLS as the First End Product

**Goal:** Stable Lawson-Hanson reproduction.
**Features:** QR-based solution, anti-cycling rules, defined tolerances, deterministic behavior, in-place & out-of-place API.
**Restrictions:** No sparse support, no AD integration, no parallelization.
**Release Criterion:** Independently robust and usable.

#### Phase 2 – QP-Subproblem Reproduction

**Goal:** Exact QP → LDP → NNLS transformation as per the reference SLSQP.
**Includes:** Linearized constraints, Hessian + gradient construction, Householder/Cholesky steps, primal recovery.
**Validation:** Comparison with NLopt results.
**Optional Intermediate:** QP-Toolkit (if clearly independent).

#### Phase 3 – Complete SQP Loop

**Goal:** Algorithmically complete SLSQP.
**Includes:** BFGS update, merit function, line-search according to reference, active-set management, termination logic.
**Test Basis:** Hock-Schittkowski, degenerate constraints, ill-conditioned cases.
**Checkpoint:** Numerical equivalence achieved. Only here is SLSQP officially "reproduced."

#### Phase 4 – Integration

Integration of SciMLBase interface, `Optimization.jl` compatibility, documentation, benchmarks, and real-world problem examples. **No algorithmic intervention.**

#### Phase 5 – Targeted Modernization (Optional)

Only after secured reproduction:

* Improved AD support.
* Alternative QP backends.
* Performance tuning.
* Sparse matrix experiments.
* Diagnostic extensions.
**Modernization is kept separate and versioned.**

---

### 4. The "Not-To-Do" List (Protective Wall until Phase 3)

Until Phase 3 is completed, the following are **prohibited**:

* No AD dispatch in the core.
* No sparse matrix support.
* No multi-threading within the solver.
* No alternative line-search strategies.
* No alternative QP backends.
* No algorithmic "improvements."
* No SHGO-specific special paths.

**Only implementation improvements are allowed—no algorithmic changes.**

---

### 5. Risk Management

* **R1 – Architectural Euphoria:** Core first, modularization second.
* **R2 – Numerical Divergences:** Reference comparison at the iteration level.
* **R3 – Scope Explosion:** Feature freeze until Phase 3 is completed.
* **R4 – Maintenance Overload:** Intermediate products are only released if independently viable.

---

### 6. Success Criteria

The project is successful if:

1. A pure Julia SLSQP exists.
2. It corresponds numerically to NLopt/SciPy.
3. It is integrable into `Optimization.jl`.
4. At least one intermediate product is independently usable.
5. Modernization occurs consciously and controlled.

---

### 7. Strategic Essence

This project follows a clear order:
**Reproduction → Stabilization → Integration → Modularization → Modernization**

SLSQP is the end product. Intermediate products are intentionally crafted building blocks. Innovation begins only after secured equivalence.

---
# SLSQP.jl – Strategic Development Paper

## Phase 0 – Reference Forensics

**Overall Goal:**  
Secure an exact, unambiguous algorithmic specification of SLSQP as the foundation for faithful reproduction.

**Scope:**  
Analysis of the original Kraft Fortran code, NLopt C-port, SciPy port, and related primary sources.

**Sub-phases (structured & sequential):**

### 0.1 – Reference Archive & Cartography
- Collect and organize all relevant primary sources, papers, code repositories, and documentation.  
- Output: Comprehensive Master Reference List (currently version 9.0).  
# 🧭 Phase 0.1 – Document Final: THE SLSQP CARTOGRAPHY

**Strategic Master Reference List 9.0 (Comprehensive Version)**
**Status:** Total Dimensional Capture Completed – No Omissions.

---

## I. Mathematical Foundations & Primary Sources

*The theoretical "Ground Truth." These documents explain the 'why' behind the numerical choices.*

1. **Kraft, D. (1988): A Software Package for Sequential Quadratic Programming**
* [DFVLR-FB 88-28 PDF](http://degenerateconic.com/uploads/2018/03/DFVLR_FB_88_28.pdf)
* **Description:** The foundational specification. It describes the LDP (Least Distance Programming) transformation and the L1-merit function. This is the ultimate reference for the core algorithm's mathematical intent.


2. **Netlib TOMS 733: The Original Kraft Code (1994)**
* [Official ACM TOMS 733 (Netlib)](https://www.netlib.org/toms/733)
* **Description:** The official ACM registration. By downloading the `tar.gz`, we obtain the **purest form of the 1988/1994 code**, untouched by SciPy or NLopt modifications. It is the forensic baseline for memory work-arrays.


3. **Kraft, D. (1994): TOMP—Fortran modules for optimal control**
* [ACM TOMS Link](https://dl.acm.org/doi/10.1145/192115.192124)
* **Description:** An extension of SLSQP into optimal control contexts. It illustrates how the algorithm was modularized in Fortran to handle structured constraints in robotics.


4. **Lawson, C. L. & Hanson, R. J. (1974/1995): Solving Least Squares Problems**
* [SIAM Classics](https://epubs.siam.org/doi/book/10.1137/1.9781611971217)
* **Description:** The fundamental text for the NNLS (Non-Negative Least Squares) sub-problem. It provides the proof for the active-set mechanics and the anti-cycling rules that prevent infinite loops.


5. **Powell, M. J. D. (1978): A Fast Algorithm for Nonlinearly Constrained Optimization**
* [Lecture Notes](https://link.springer.com/chapter/10.1007/BFb0067703)
* **Description:** The origin of the Damped BFGS update. This paper explains how to maintain a positive definite Hessian approximation even when the objective is not globally convex.


6. **Schittkowski, K. (1982): On the convergence of SQP methods**
* [Mathematical Programming Study 18](https://link.springer.com/book/10.1007/978-3-642-48320-2)
* **Description:** A deep dive into the augmented Lagrangian line search function used in SQP. It provides the convergence proofs that justify Kraft's merit function choices.


7. **Bemporad, A. (2016): A Non-Negative Least Squares Algorithm for QP**
* [ArXiv 1510.06202](https://arxiv.org/abs/1510.06202)
* **Description:** A modern analysis showing how general QPs can be solved as NNLS problems. This is a crucial link for making Phase 2 (QP-subproblem) numerically robust.



---

## II. Reference Implementations (The Evolution of Code)

*How the theory was translated into different languages. Essential for the "Despaghettisierung".*

8. **Jacob Williams: Modern Fortran SLSQP (2008+)**
* [GitHub - jacobwilliams/slsqp](https://github.com/jacobwilliams/slsqp)
* **Description:** This is a modernized, object-oriented re-implementation of the Kraft code. It is **extremely valuable** because it shows a structured approach to the legacy code, helping us map Fortran work-arrays to modern Julia types.


9. **NLopt: slsqp.c (The C Reference)**
* [GitHub Source](https://github.com/stevengj/nlopt/blob/master/src/algs/slsqp/slsqp.c)
* **Description:** Steven G. Johnson’s C-port. This is the most widely used version in the Julia ecosystem today. It contains vital fixes for uninitialized variables found in the original Fortran.


10. **SciPy: slsqp_opt.f (The Python Engine)**
* [GitHub Source](https://github.com/scipy/scipy/blob/master/scipy/optimize/slsqp/slsqp_opt.f)
* **Description:** The version used by millions of Python users. Analyzing the Python-to-Fortran wrapper (`slsqp.pyf`) shows how the community handles user-defined tolerances.


11. **Relf: Rust SLSQP**
* [GitHub relf/slsqp](https://github.com/relf/slsqp)
* **Description:** A Rust implementation generated via `c2rust` from NLopt. It serves as a benchmark for how other safe languages have handled the porting process.


12. **PySLSQP (2024): Transparent Logging**
* [GitHub Yosef-Guevara/PySLSQP](https://github.com/Yosef-Guevara/PySLSQP)
* **Description:** A modern Python wrapper focused on visibility. It allows extracting merit function values and constraint violations at every iteration—features we want to build into SLSQP.jl by default.



---

## III. The Julia Ecosystem (Integration Points)

*Where our new solver will live and what it will use.*

13. **NonNegLeastSquares.jl (v0.5.0, 2026)**
* [GitHub JuliaLinearAlgebra](https://github.com/JuliaLinearAlgebra/NonNegLeastSquares.jl)
* **Description:** The most up-to-date Julia implementation of NNLS with **Automatic Differentiation (AD) support**. This is a likely candidate for our Phase 1 foundation.


14. **NNLS.jl (rdeits)**
* [GitHub rdeits/NNLS.jl](https://github.com/rdeits/NNLS.jl)
* **Description:** An older, battle-tested Julia port of the Lawson-Hanson code. Valuable for cross-checking numerical consistency with legacy results.


15. **Optimization.jl (SciML)**
* [GitHub SciML/Optimization.jl](https://github.com/SciML/Optimization.jl)
* **Description:** The primary interface. Our solver must implement the `AbstractOptimizer` traits defined here to ensure community adoption.


16. **NLPModels.jl (JuliaSmoothOptimizers)**
* [GitHub JSO](https://github.com/JuliaSmoothOptimizers/NLPModels.jl)
* **Description:** Standardized modeling infrastructure. Integrating with JSO allows us to use their extensive benchmarking tools.


17. **SqpSolver.jl (Exanauts)**
* [GitHub exanauts/SqpSolver.jl](https://github.com/exanauts/SqpSolver.jl)
* **Description:** A different Julia-based SQP approach. Studying their code helps us identify "Julia-native" pitfalls in SQP implementation.



---

## IV. Testing, Benchmarks & Pathologies

*The "Judges." How we prove our implementation is correct.*

18. **SciPy SLSQP Test Suite (Numerical Benchmarks)**
* [GitHub test_slsqp.py](https://github.com/scipy/scipy/blob/main/scipy/optimize/tests/test_slsqp.py)
* **Description:** **Critical Reference.** This contains explicit comparison tests against known Hock-Schittkowski solutions. We will use these to verify numerical equivalence in Phase 3.


19. **Hock-Schittkowski Collection (HS)**
* [HS Test Problems](http://www.ai7.uni-bayreuth.de/test_problems.htm)
* **Description:** The gold standard for nonlinear optimization testing. 306 problems with documented optimal solutions.


20. **I-SLSQP (2024): Improved SLSQP**
* [ArXiv 2402.10396](https://arxiv.org/abs/2402.10396)
* **Description:** Recent research highlighting stability fixes for the dual-to-primal recovery. Essential for our Phase 5 "Modernization" goals.


21. **The Maratos Effect (Failure Case)**
* [Wikipedia Background](https://en.wikipedia.org/wiki/Maratos_effect)
* **Description:** A theoretical "trap" where SQP steps are rejected despite being good. We need this to test the robustness of our merit function logic.


22. **NLopt Issue Tracker (SLSQP label)**
* [GitHub NLopt Issues](https://www.google.com/search?q=https://github.com/stevengj/nlopt/issues%3Fq%3Dis%253Aissue%2Bslsqp)
* **Description:** Real-world error reports. Provides a list of edge cases (e.g., singular constraints) that we must handle more gracefully than the C-version.



---

## V. Advanced Dimensions (AD & Hardware)

*Future-proofing for Phase 4 and 5.*

23. **Enzyme.jl (High-Performance AD)**
* [Enzyme Documentation](https://www.google.com/search?q=https://enzyme.mit.edu/julia/)
* **Description:** Necessary for differentiating through the mutating, in-place code we will write for SLSQP.jl.


24. **Parallel-QR in SQP (2025)**
* [ACM Euro-Par 2025](https://dl.acm.org/doi/10.1007/978-3-031-99872-0_15)
* **Description:** Research on accelerating the sub-problems using parallel linear algebra. Relevant for scaling the solver to larger dimensions.
**End of Phase 0.1.**


- Status: Completed.

### 0.2 – Numerical Constants & Tolerances Table
- Extract all hard-coded numerical values, tolerances, and thresholds.  
- Output: Table with columns: Parameter, Value, Source (with line/reference), Purpose, Sensitivity/Notes.

### Phase 0.2 – Numerical Constants & Tolerances Table  
**Version:** 1.3 (improved, categorized & consolidated)  
**Status:** Freeze-ready after forensics  

**Approach**  
- Primary sources: NLopt `slsqp.c` (main reference), SciPy `slsqp_opt.f` + Python wrapper, Kraft DFVLR-FB 88-28 (original intent).  
- Method: Manual extraction + cross-check with SciPy test suite & NLopt issues.  
- Improvements:  
  - Categorized into 3 blocks for clarity (core, termination, safety).  
  - Added missing "ghost constants" (curvature safeguard, activation threshold, and line-search factors).  
  - Added interaction matrix (small, 8 entries) to show dependencies.  
  - Noted SciPy vs NLopt divergences with "Reference Mode" suggestion (strict_nlopt / strict_scipy for testing).  
- Criteria: Only values influencing termination, feasibility, damping, or stability; Julia notes for adaptation (e.g. eps(T)).  

#### A. Algorithm Core Parameters (not changeable until Phase 4)
These are mathematical invariants – do not override.

| #  | Parameter        | Default Value | Source / Location                  | Purpose / Usage                                      | Sensitivity / Julia Note                                                                 |
|----|------------------|---------------|------------------------------------|------------------------------------------------------|------------------------------------------------------------------------------------------|
| 1  | `rho_init`       | 1.0          | Kraft p. 12 + slsqp.c              | Initial L1 penalty parameter                         | Medium; dynamic increase. Julia: Keep fixed.                                             |
| 2  | `rho_factor`     | 10.0         | slsqp.c (typisch)                  | Multiplier for penalty increase                      | High; aggressiveness factor. SciPy sometimes 100 – note in 0.6.                          |
| 3  | `theta_lim`      | 0.2          | Powell 1978 + slsqp.c BFGS-damping | Threshold for damped BFGS update                     | High; prevents negative curvature. Julia: Keep fixed.                                    |
| 4  | `sigma`          | 0.1          | Kraft p. 14 + SciPy backtracking   | Contraction factor in line search (Armijo)           | Medium; step *= sigma on failure. Classic range 0.1–0.5.                                |
| 5  | `eta`            | 0.01         | Kraft / NLopt Armijo-Goldstein     | Sufficient decrease parameter                        | High; Armijo condition: f(x+αd) ≤ f(x) + ηα ∇f·d. Julia: Keep fixed.                    |

#### B. Termination & User-Facing Controls (user-overridable)
These can be parameters in the API.

| #  | Parameter          | Default Value | Source / Location                  | Purpose / Usage                                      | Sensitivity / Julia Note                                                                 |
|----|--------------------|---------------|------------------------------------|------------------------------------------------------|------------------------------------------------------------------------------------------|
| 6  | `acc` / `ACC`      | 1.0e-6        | SciPy wrapper + slsqp_opt.f        | General convergence accuracy / termination tol       | High; SciPy 1e-6 vs NLopt 1e-8 (divergence risk). Julia: User-option + relative scaling. |
| 7  | `maxiter`          | 1000 or 3*n   | slsqp.c + Kraft                    | Maximum major iterations                             | Low; NLopt 1000, SciPy 100*n. Julia: User-default 3*n.                                  |
| 8  | `maxfun`           | 1000 or 10*n  | slsqp_opt.f & NLopt                | Maximum function evaluations                         | Low; Prevents loops. Julia: User-default 10*n.                                           |
| 9  | `ftol_rel` / `abs` | 1e-8 / 1e-10  | NLopt & SciPy termination          | Relative / absolute objective change                 | High; Termination if |f_k - f_{k-1}| < ftol_rel * |f| + ftol_abs. Julia: eps(T)-scaled. |
| 10 | `xtol_rel` / `abs` | 1e-8 / 1e-10  | NLopt & SciPy termination          | Relative / absolute variable change                  | High; Termination if ||x_k - x_{k-1}|| < xtol_rel * ||x|| + xtol_abs. Julia: eps(T).   |
| 11 | `constr_viol_tol`  | 1e-8          | Kraft & NLopt feasibility          | Acceptable constraint violation after step           | High; Often same as tol. Julia: eps(T)-scaled.                                           |

#### C. Numerical Safety Guards (adaptable in Julia)
These are numerical protections – can be Julia-optimized.

| #  | Parameter            | Default Value | Source / Location                  | Purpose / Usage                                      | Sensitivity / Julia Note                                                                 |
|----|----------------------|---------------|------------------------------------|------------------------------------------------------|------------------------------------------------------------------------------------------|
| 12 | `eps` / `EPS`        | ~1e-7..1e-8   | slsqp.c (machine eps impl)         | Rank detection, zero checks, singular matrix         | High; Not fixed! Julia: Use `sqrt(eps(T))` or `eps(T)` (Float64 ~1e-8).                  |
| 13 | `w_tol` (NNLS dual)  | 1e-8          | slsqp.c + Lawson-Hanson            | Dual feasibility in NNLS (w_max <= w_tol → optimal)  | High; Affects active-set decisions. Julia: eps(T).                                       |
| 14 | `alpha_min`          | 1e-10         | slsqp.c line search                | Minimum step size (backtracking abort)               | Low; Prevents underflow. Julia: nextfloat(zero(T)).                                      |
| 15 | `tiny` / `small`     | 1e-30..1e-40  | slsqp_opt.f + slsqp.c              | Lower bound for divisions / denominators             | Low; NaN/Inf safety. Julia: `eps(T)^2`.                                                  |
| 16 | `curvature_guard`    | 1e-10         | Powell + slsqp.c BFGS              | Absolute threshold for BFGS skip (if s'y < guard)    | Medium; Prevents bad updates. Julia: eps(T).                                             |
| 17 | `activation_thresh`  | eps or 0      | Lawson-Hanson + slsqp.c            | Constraint activation (if abs(x) < thresh → active)  | High; Often <=0 or <eps. Julia: eps(T).                                                  |

### Interaction Matrix (small, as suggested)
Shows direct dependencies between parameters (critical for understanding divergences).

| Interaction               | Effect / Interplay                                 | Sensitivity / Note |
|---------------------------|----------------------------------------------------|--------------------|
| `acc ↔ constr_viol_tol`   | Feasible termination (viol < constr_viol_tol)      | High; SciPy vs NLopt differ – reference mode needed |
| `eps ↔ rank test`         | QP solvability (if abs(r) < eps*norm(A) → singular)| High; Affects degenerate constraints |
| `rho_init ↔ eta`          | Step rejection (sufficient decrease in merit)      | Medium; Armijo interacts with penalty |
| `theta_lim ↔ curvature_guard` | BFGS skip frequency (s'y < guard or < theta_lim) | High; Prevents ill-conditioned B |
| `w_tol ↔ tol`             | NNLS → overall convergence (dual tol feeds KKT)    | High; Often same value |
| `alpha_min ↔ tiny`        | Line search abort (step < alpha_min → fail)        | Low; Numerical safety net |
| `maxiter ↔ maxfun`        | Early termination (if maxfun reached first)        | Low; Protects expensive functions |
| `ftol_rel ↔ xtol_rel`     | Balanced stop (objective vs variable change)       | Medium; User-overridable |

**End of Phase 0.2.**  


- Status: Pending – next immediate step.

### 0.3 – High-Level Julia-like Pseudocode
- Create simplified, Julia-style pseudocode representations of the main control flows.  
- Focus: Overall SQP loop, NNLS outer/inner loop, QP → LDP → NNLS transformation.  
- Output: 3–5 annotated pseudocode blocks (Markdown).  
- Status: Pending.

### 0.4 – Decision Logic & Exit Conditions
- Document all conditional branches, mode switches, exit codes, and reset conditions.  
- Output: Table or structured list (Condition → Action → Source → Meaning).  
- Status: Pending.

### 0.5 – Pathologies Mapping + First Draft of Architecture
- Map known failure modes, instabilities, and edge cases from literature and implementations.  
- Simultaneously produce the first rough architecture proposal (Proposal 1.0).  
- Output:  
  - Short pathologies table (Problem → Symptom → Original handling → Reference)  
  - Architecture Proposal 1.0 (very slim draft: product line, workspace sketch, principles)
  - 
## Phase 0.5 – First Draft of Architecture
**Architecture Proposal 1.0 (Preliminary Sketch)**  
**Status:** Preliminary first draft — created after Phase 0.1  
**Purpose:** Provide rough direction; to be validated and refined in Phase 1

### 1. Overall Structure (very high-level)

- Package name (provisional): **SLSQP.jl**  
- Primary goal: a faithful SLSQP solver  
- Planned independent modules (in order of development):  
  1. **CoreNNLS.jl** (Phase 1 – universally reusable)  
  2. **QPTransform.jl** (Phase 2)  
  3. **SLSQP.jl** (Phase 3 – integration of the above)

### 2. Central Architectural Idea (the only fixed element so far)

All modules will share a **mutable workspace pattern** to enable zero-allocation loops and type stability.

**Preliminary conceptual workspace outline (sketch only – not final):**

```julia
struct SLSQPWorkspace{T<:AbstractFloat}
    x::Vector{T}           # current point
    g::Vector{T}           # gradient
    c::Vector{T}           # constraints
    lambda::Vector{T}      # multipliers
    B::Matrix{T}           # Hessian approximation
    d::Vector{T}           # search direction
    # ... to be extended later
end
```

### 3. Core Guiding Principles (short & unchanged)

- Reproduction before innovation  
- In-place operations and type-generic design from the start  
- SciMLBase-compatible (`AbstractOptimizer`)  
- No new algorithms or features before Phase 4

**End of Phase 0.5.**
- Status: Architecture draft created (Phase 0.5 section already available).

### 0.6 – Go/No-Go & Phase 0 Closure
- Summarize all outputs from 0.1–0.5.  
- Evaluate readiness for Phase 1 (CoreNNLS).  
- Output: Final Phase 0 Report (Markdown/PDF) + explicit Go/No-Go decision.  
- Status: Pending – final checkpoint of Phase 0.

**Phase 0 Duration Estimate:** 2–4 weeks (pragmatic execution).  
**Success Criterion for Phase 0:**  
We have a complete, traceable foundation that allows faithful reproduction without guessing.




---



