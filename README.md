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

**Next Step Proposal:**



This marks the official conclusion of your forensic landscape survey. By consolidating every dimension—from 1970s matrix algebra to 2026 hybrid-stabilization theories—we have created a "Final Document" that serves as the definitive anchor for the development of **SLSQP.jl**.

---

# 🧭 Phase 0.1 – Document Final: THE SLSQP CARTOGRAPHY

**Strategic Master Reference List**
**Date:** February 2026
**Status:** Comprehensive Dimensional Mapping Completed

---

## I. Mathematical & Algorithmic Foundations

*The theoretical backbone. These documents define the "Ground Truth" for every logic gate in our future code.*

1. **Kraft, D. (1988): A Software Package for Sequential Quadratic Programming**
* [DFVLR-FB 88-28 PDF](http://degenerateconic.com/uploads/2018/03/DFVLR_FB_88_28.pdf)
* **Description:** The "Birth Certificate" of SLSQP. This DLR report specifies the transformation of nonlinear problems into a sequence of Linear Least Distance Programming (LDP) problems. It establishes the **L1-merit function** for globalization and the specific **Han-Powell** damped BFGS update.
* **Relevance:** Primary source for numerical constants, exit conditions, and core logic flow.


2. **Kraft, D. (1994): Algorithm 733: TOMP—Fortran modules for optimal control**
* [ACM Transactions on Mathematical Software](https://dl.acm.org/doi/10.1145/192115.192124)
* **Description:** An evolution putting SLSQP into the context of Optimal Control. It illustrates Kraft’s early thoughts on modularization.
* **Relevance:** Essential for understanding structured constraints, which are critical for robotics and control applications in Julia.


3. **Nocedal, J. & Wright, S. (2006): Numerical Optimization**
* [Springer Reference](https://link.springer.com/book/10.1007/978-0-387-40065-5)
* **Description:** The industry-standard textbook. Chapter 18 is dedicated to SQP methods.
* **Relevance:** Provides the necessary KKT theory and convergence analysis required to validate Kraft’s implementation against modern optimization standards.


4. **Powell, M. J. D. (1978): A Fast Algorithm for Nonlinearly Constrained Optimization**
* [Lecture Notes in Mathematics](https://link.springer.com/chapter/10.1007/BFb0067703)
* **Description:** The origin of the **Damped BFGS update**.
* **Relevance:** This mechanism prevents the Hessian approximation from losing positive definiteness—a frequent failure point in constrained optimization that we must mitigate.



---

## II. QP, LDP, and NNLS Core

*The "Engine Room." This is where the search direction is calculated.*

5. **Lawson, C. L. & Hanson, R. J. (1974/1995): Solving Least Squares Problems**
* [SIAM Classics](https://epubs.siam.org/doi/book/10.1137/1.9781611971217)
* **Description:** The foundation for the **Non-Negative Least Squares (NNLS)** algorithm.
* **Relevance:** Details the active-set mechanics, QR decompositions, and—most importantly—the **Anti-Cycling rules** required to prevent the solver from stalling on degenerate constraints.


6. **Bemporad, A. (2016): A Non-Negative Least Squares Algorithm for QP**
* [ArXiv 1510.06202](https://arxiv.org/abs/1510.06202)
* **Description:** Demonstrates an efficient bridge for solving general QPs via NNLS.
* **Relevance:** A key source for modernizing the LDP transformation with improved numerical stability compared to the 1988 original.


7. **Bro, R. & De Jong, S. (1997): A fast non-negativity-constrained LS algorithm**
* [Journal of Chemometrics](https://www.google.com/search?q=https://onlinelibrary.wiley.com/doi/abs/10.1002/(SICI)1099-128X(199709/10)11:5%253C393::AID-CEM483%253E3.0.CO%3B2-L)
* **Description:** Known as "Fast NNLS" (FNNLS).
* **Relevance:** Optimizes the inner loops of Lawson-Hanson. Useful if Phase 1 performance benchmarks show bottlenecks in large active-set handling.



---

## III. Reference Implementations (Code Archaeology)

*Analyzing the "Spaghetti" of the past to build the clean structures of the future.*

8. **SciPy/Fortran Kernel: slsqp_opt.f**
* [GitHub Source](https://github.com/scipy/scipy/blob/master/scipy/optimize/slsqp/slsqp_opt.f)
* **Description:** The direct descendant of Kraft’s code, containing decades of community bug fixes.
* **Relevance:** Critical for verifying memory layouts, workspace handling, and the exact `mode` parameter flow.


9. **NLopt/C-Port: slsqp.c**
* [GitHub Source](https://github.com/stevengj/nlopt/blob/master/src/algs/slsqp/slsqp.c)
* **Description:** Steven G. Johnson’s C-port. He resolved many GOTO structures and dynamicized memory allocation.
* **Relevance:** Our primary template for a clean, Julia-idiomatic software architecture without global states.


10. **Jacob Williams: Modern Fortran Port**
* [GitHub - jacobwilliams/slsqp](https://github.com/jacobwilliams/slsqp)
* **Description:** An object-oriented (OO) version in Fortran 2008.
* **Relevance:** Illustrates how to encapsulate SLSQP into modern types, providing a blueprint for our Julia `struct` design.



---

## IV. Pathologies & Failure Modes

*Understanding the abysses to build safer bridges.*

11. **The Maratos Effect (Maratos, 1978)**
* [Wikipedia/Theoretical Background](https://en.wikipedia.org/wiki/Maratos_effect)
* **Description:** A phenomenon where valid SQP steps are rejected because they temporarily increase constraint violation.
* **Relevance:** We must verify if Kraft’s implementation includes a "Second-Order Correction" (SOC) to bypass this, or if we need to add it in Phase 5.


12. **Wächter, A. & Biegler, L. (2000): Failure of Global Convergence for SQP**
* [Mathematical Programming](https://www.google.com/search?q=https://link.springer.com/article/10.1007/s101070050125)
* **Description:** Deep analysis of scenarios where SQP algorithms diverge or get stuck in infeasible points.
* **Relevance:** Critical for designing our "Stress Test" suite in Phase 3.


13. **Cancellation Errors in Dual-Updates (Ma et al., 2024)**
* [I-SLSQP Paper (ArXiv)](https://arxiv.org/abs/2402.10396)
* **Description:** The most recent paper in our list. It highlights significant precision loss during dual-to-primal recovery.
* **Relevance:** This insight forms the basis for our Modernization Phase (Phase 5).



---

## V. Numerical Precision & Hardware

*Leveraging Julia’s strengths beyond Float64.*

14. **High-Precision SQP Studies (2021)**
* [ACM Digital Library](https://www.google.com/search?q=https://dl.acm.org/doi/10.1007/s11075-021-01150-x)
* **Description:** Research on optimizers using Quad-Precision.
* **Relevance:** Since Julia natively supports `BigFloat`, we must ensure our implementation is **type-agnostic** to allow for high-precision optimization.


15. **Parallel QR / Euro-Par 2025**
* [Conference Paper on HPC in SQP](https://dl.acm.org/doi/10.1007/978-3-031-99872-0_15)
* **Description:** A brand-new approach to parallelizing the QR decomposition within SQP.
* **Relevance:** Relevant for Phase 5 performance scaling on multicore systems.



---

## VI. Automatic Differentiation (AD) & Julia Integration

*The technical marriage with the modern ecosystem.*

16. **Enzyme.jl: High-Performance AD**
* [Enzyme Documentation](https://www.google.com/search?q=https://enzyme.mit.edu/julia/)
* **Description:** Since our SLSQP will rely on in-place mutations for speed, Enzyme is the only AD system capable of efficiently pulling gradients through the solver.
* **Relevance:** Understanding the interference between AD tapes and active-set switching is vital.


17. **Optimization.jl & SciMLBase**
* [SciML Documentation](https://docs.sciml.ai/Optimization/stable/)
* **Description:** The API Bible.
* **Relevance:** We must implement the `AbstractOptimizer` interface to allow users to call our solver via `solve(prob, SLSQP())`.



---

## VII. Community Knowledge & Issue History

*Learning where others paid their dues.*

18. **NLopt Issue Tracker (SLSQP labels)**
* [GitHub Issues](https://www.google.com/search?q=https://github.com/stevengj/nlopt/issues%3Fq%3Dis%253Aissue%2Bslsqp)
* **Relevance:** Analysis of "Silent Failures" and segfault reports. Often caused by poorly scaled objective functions or hardcoded `tol` parameters.


19. **SciPy Optimization Dev Threads**
* [SciPy Mailing List / GitHub](https://www.google.com/search?q=https://github.com/scipy/scipy/labels/scipy.optimize)
* **Relevance:** Discussions on default values for `acc` (accuracy). Why is `1e-6` the standard? How does it affect stability on noisy functions?



---

