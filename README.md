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



---

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


**Phase 0.1 – Document Final** ist hiermit abgeschlossen.
