
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

# 🧭 Phase 0.1 – THE SLSQP STRATEGIC CARTOGRAPHY (Expanded Reconciliation Edition)

**Version:** 13.5 – Final Corrected  
**Date:** February 12, 2026  
**Status:** All 28 unique references reconciled and numbered sequentially (1–28). No omissions. No duplicate numbering.

This document defines the authoritative cartography of sources governing the faithful reconstruction and controlled evolution of SLSQP.

---

# LEVEL R — RECONSTRUCTION REFERENCE

*(Binding for Phases 0–3. All numerical behavior and control flow must trace back exclusively to these sources.)*

This level defines the canonical DNA of SLSQP.  
No modernization, reinterpretation, or structural deviation is permitted during reconstruction.

---

## R1 — Mathematical Foundations (The Immutable Core)

**1. Dieter Kraft (1988)**  
*A Software Package for Sequential Quadratic Programming* (DFVLR-FB 88-28)  
[PDF (Stable Mirror)](http://degenerateconic.com/uploads/2018/03/DFVLR_FB_88_28.pdf)  
**Role:** Foundational mathematical specification.  
**Binding for:** L1 merit function, QP → LDP transformation, penalty parameter logic.

**2. Netlib – TOMS 733 (1994)**  
[Netlib Archive](https://netlib.org/toms/733)  
**Role:** Original Fortran implementation.  
**Binding for:** Work-array layout, memory indexing, MODE return codes, major/minor loop structure.

**3. Dieter Kraft (1994): TOMP — Fortran modules for optimal control**  
[ACM TOMS DOI](https://dl.acm.org/doi/10.1145/192115.192124)  
**Role:** Modular extension of SLSQP.  
**Binding for:** Structured constraints handling.

**4. Charles L. Lawson & Richard J. Hanson (1974/1995)**  
*Solving Least Squares Problems*  
[SIAM Classics](https://epubs.siam.org/doi/book/10.1137/1.9781611971217)  
**Role:** NNLS algorithm (Chapter 23).  
**Binding for:** Active-set mechanics, dual feasibility tests, anti-cycling safeguards.

**5. Michael J. D. Powell (1978)**  
*A Fast Algorithm for Nonlinearly Constrained Optimization Calculations*  
[Springer DOI](https://doi.org/10.1007/BFb0067703)  
**Role:** Damped BFGS update mechanism.  
**Binding for:** `theta_lim` safeguard, curvature condition handling, positive definiteness maintenance.

**6. Klaus Schittkowski (1983)**  
*On the Convergence of a Sequential Quadratic Programming Method with an Augmented Lagrangian Line Search Function*  
[ResearchGate PDF](https://www.researchgate.net/publication/233116651_On_the_convergence_of_a_sequential_quadratic_programming_method_with_an_augmented_Lagrangian_line_search_function)  
**Role:** Convergence analysis.  
**Binding for:** Theoretical justification of global convergence strategy.

**7. Alberto Bemporad (2016)**  
*A Quadratic Programming Algorithm Based on Nonnegative Least Squares With Applications to Embedded Model Predictive Control*  
[IEEE TAC PDF](http://cse.lab.imtlucca.it/~bemporad/publications/papers/ieeetac-qpnnls.pdf)  
**Role:** Robust QP-to-NNLS linkage.  
**Binding for:** Phase 2 transformation stability.

---

## R2 — Reference Implementations (Forensic Cross-Validation)

**8. Jacob Williams: Modern Fortran SLSQP (2008–present)**  
[GitHub – jacobwilliams/slsqp](https://github.com/jacobwilliams/slsqp)  
**Role:** Object-oriented refactoring reference.  
**Status:** Structural clarity reference.

**9. NLopt: slsqp.c (C Reference)**  
[GitHub Source](https://github.com/stevengj/nlopt/blob/master/src/algs/slsqp/slsqp.c)  
**Role:** Stability fixes and edge-case handling.  
**Status:** Secondary forensic comparator.

**10. SciPy: SLSQP Fortran Core and Python Wrapper**  
[GitHub Source – _slsqp.py](https://github.com/scipy/scipy/blob/main/scipy/optimize/_slsqp.py)  
**Role:** Tolerance semantics and convergence defaults.  
**Status:** Behavioral validation target.

**11. relf/slsqp: Rust Port via c2rust**  
[GitHub – relf/slsqp](https://github.com/relf/slsqp)  
**Role:** Memory-safe translation benchmark.  
**Status:** Optional cross-validation for porting issues.

**12. PySLSQP (2024–present): Transparent Python Package**  
[GitHub – anugrahjo/PySLSQP](https://github.com/anugrahjo/PySLSQP)  
**Role:** Logging and diagnostics wrapper.  
**Status:** Inspiration for Phase 5 diagnostics.

---

## R3 — Validation & Benchmark Authority

**13. Hock–Schittkowski Test Collection (HS 1–306)**  
[Official PDF](http://klaus-schittkowski.de/test_problems.pdf)  
**Role:** Numerical equivalence and degeneracy tests.  
**Status:** Binding for Phase 3 validation.

**14. SciPy SLSQP Test Suite**  
[GitHub – test_slsqp.py](https://github.com/scipy/scipy/blob/main/scipy/optimize/tests/test_slsqp.py)  
**Role:** Regression validation.  
**Status:** Binding for regression testing.

---

## 🔒 Reconstruction Axiom

During Phases 0–3:  
- No algorithmic reinterpretation  
- No sparse reformulation  
- No AD integration  
- No merit-function redesign  

Every non-trivial numerical decision must be traceable to a documented reference in LEVEL R.

---

# LEVEL E — EVOLUTION REGISTER

*(Non-binding. Strategic outlook for Phase 5 and beyond. Must not influence reconstruction decisions.)*

---

## E1 — Stability Extensions

**15. OpenSQP (2025): Reconfigurable Open-Source SQP in Python**  
[arXiv 2512.05392](https://arxiv.org/abs/2512.05392)  
**Purpose:** Reconfigurable variants for robustness.

**16. I-SLSQP (2024): Improved SLSQP with Stability Fixes**  
[arXiv 2402.10396](https://arxiv.org/abs/2402.10396)  
**Purpose:** Dual-to-primal recovery enhancements.

---

## E2 — Performance & Scalability

**17. Efficient Task Graph Scheduling for Parallel QR Factorization in SLSQP (2025)**  
[arXiv 2506.09463](https://arxiv.org/abs/2506.09463)  
**Purpose:** Parallel QR for speedup.

---

## E3 — Automatic Differentiation Pathways

**18. Enzyme.jl – High-performance automatic differentiation**  
[GitHub – EnzymeAD/Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl)  
**Purpose:** Mutating code differentiation (relevant for differentiating the solver itself).

---

## E4 — Julia Ecosystem Integration

**19. NonNegLeastSquares.jl**  
[GitHub – JuliaLinearAlgebra](https://github.com/JuliaLinearAlgebra/NonNegLeastSquares.jl)  
**Purpose:** NNLS with AD potential (Candidate for Phase 1 foundation).

**20. NNLS.jl (Rob Deits)**  
[GitHub – rdeits/NNLS.jl](https://github.com/rdeits/NNLS.jl)  
**Purpose:** Lawson–Hanson port (Supplemental cross-check to #19).

**21. Optimization.jl (SciML ecosystem)**  
[GitHub – SciML/Optimization.jl](https://github.com/SciML/Optimization.jl)  
**Purpose:** Unified interface (Target for Phase 4 integration).

**22. NLPModels.jl (JuliaSmoothOptimizers)**  
[GitHub – JuliaSmoothOptimizers/NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl)  
**Purpose:** Modeling and benchmarks.

**23. SqpSolver.jl**  
[GitHub – exanauts/SqpSolver.jl](https://github.com/exanauts/SqpSolver.jl)  
**Purpose:** Alternative SQP implementation reference.

**24. Nonconvex.jl**  
[GitHub – JuliaNonconvex/Nonconvex.jl](https://github.com/JuliaNonconvex/Nonconvex.jl)  
**Purpose:** Non-convex toolbox (Supplemental).

**25. Optim.jl**  
[GitHub – JuliaNLSolvers/Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)  
**Purpose:** Unconstrained optimization complements.

---

## E5 — Pathologies & Testing Extensions

**26. The Maratos Effect**  
[Wikipedia Entry](https://en.wikipedia.org/wiki/Maratos_effect)  
**Purpose:** Merit function robustness test.

**27. NLopt SLSQP Issue Tracker**  
[GitHub Issues – search “slsqp”](https://github.com/stevengj/nlopt/issues?q=is%3Aissue+is%3Aopen+slsqp)  
**Purpose:** Real-world edge cases.

**28. Fast and Robust Least Squares / Curve Fitting in Julia (JuliaCon 2025)**  
[YouTube Video](https://www.youtube.com/watch?v=mdcCjaYSNNc)  
**Purpose:** Stable fitting techniques relevant to SLSQP sub-problems.

---

# End of Phase 0.1

### 0.2 – Numerical Constants & Tolerances Table

**Version:** 3.2 – Final Consolidated  
**Status:** Forensic-complete and frozen for reconstruction.  
**Language:** English (Authoritative)

This phase documents the numerical "DNA" of SLSQP. Following the Dual-Layer standard from Phase 0.1, parameters are separated into:

*   **LEVEL R (Binding Reconstruction):** Mandatory values for Phases 0–3 to ensure numerical equivalence.
*   **LEVEL E (Evolutionary/Optional):** Parameters for future adaptation (Phase 5+). These must **not** alter reconstruction behavior.

---

# LEVEL R — CANONICAL DEFAULTS (Binding)

*Mandatory values for Phases 0–3. Every non-trivial decision must trace back to this level.*

## R1 — Core Algorithmic Constants

| Parameter             | Value  | Source           | Role / Description                                                                 |
| --------------------- | ------ | ---------------- | ---------------------------------------------------------------------------------- |
| `rho_init`            | 10.0   | NLopt/SciPy      | Initial L1 penalty parameter. (Kraft suggests 1.0, but standard impl. use 10.0).   |
| `rho_factor`          | 10.0   | Kraft/NLopt      | Multiplier for penalty updates; prevents stagnation.                               |
| `delta_lambda_offset` | 1e-2   | NLopt impl       | Maratos guard margin; ensures $\rho > \|\lambda\|_\infty + \text{offset}$.          |
| `theta_lim`           | 0.2    | Powell 1978      | Damping threshold for BFGS updates ($s^Ty < \theta \cdot s^TBs$).                  |
| `curvature_guard`     | 1e-10  | slsqp.c          | Absolute guard for $s^Ty$; skips BFGS update if below threshold.                   |
| `sigma`               | 0.1    | Kraft            | Backtracking reduction factor in Armijo line search ($\alpha \leftarrow \alpha\sigma$). |
| `eta`                 | 0.01   | Kraft/NLopt      | Armijo slope parameter for sufficient merit decrease.                              |
| `alpha_min`           | 1e-10  | slsqp.c          | Minimum step size before declaring line search failure.                            |
| `alpha_ratio_tol`     | ~1e-12 | Implicit in NNLS | Boundary step tie tolerance; prevents cycling on degenerate ratios.                |

**Notes:** High sensitivity for `rho_factor`. SciPy sometimes uses 100 in wrappers. We use 10.0 for strict Kraft/NLopt equivalence.

---

## R2 — Primary Termination Criteria

| Parameter         | Value                       | Source      | Role / Description                                                               |
| ----------------- | --------------------------- | ----------- | -------------------------------------------------------------------------------- |
| `acc`             | 1e-6 (SciPy) / 1e-8 (NLopt) | wrapper     | General convergence accuracy (KKT norm / feasibility). High divergence source.   |
| `maxiter`         | 100 (SciPy) or 3*n (NLopt)  | slsqp.c     | Maximum major iterations; scales with $n$ in NLopt.                              |
| `maxfun`          | 1000 or 10*n                | slsqp_opt.f | Maximum function/gradient evaluations.                                           |
| `constr_viol_tol` | 1e-8                        | Kraft/NLopt | Feasibility tolerance ($\max(\|c\|) < \text{tol}$). High sensitivity.            |

**Notes:** SciPy defaults to tighter iteration limits for benchmarks; use problem-dependent scaling for equivalence.

---

## R3 — Numerical Safety Guards

| Parameter             | Value             | Source        | Role / Description                                                        |
| --------------------- | ----------------- | ------------- | ------------------------------------------------------------------------- |
| `eps_machine`         | machine epsilon   | C impl        | Floating-point epsilon for zero detection; Julia: `eps(T)`.               |
| `eps_rank`            | ~sqrt(eps)        | slsqp.c usage | Rank detection in QR/Cholesky (~1.49e-8 for Float64). Distinct from eps.  |
| `w_tol`               | 1e-8              | NNLS          | Dual feasibility tolerance in NNLS ($\max(w_Z) \le \text{tol}$).          |
| `activation_thresh`   | eps               | Lawson-Hanson | Threshold for constraint activation.                                      |
| `tiny`                | 1e-30             | slsqp_opt.f   | Lower bound for divisions to prevent underflow.                           |
| `B_reset_tol`         | ~1e-14 (implicit) | NLopt         | Reset Hessian $B$ to Identity if positive definiteness is lost.           |
| `max_curvature_fails` | 3                 | NLopt impl    | Threshold to reset Hessian B to I after repeated curvature violations.    |

---

# LEVEL E — EVOLUTIONARY PARAMETERS (Optional / Adaptive)

*Architecturally prepared parameters for Phase 5; do not impact Phases 0–3.*

## E1 — Advanced Convergence Controls

| Parameter  | Value | Source | Role / Description                                                |
| ---------- | ----- | ------ | ----------------------------------------------------------------- |
| `ftol_rel` | 1e-8  | NLopt  | Relative objective change ($\Delta f / |f| < \text{tol}$).        |
| `ftol_abs` | 1e-10 | NLopt  | Absolute objective change ($\Delta f < \text{tol}$).              |
| `xtol_rel` | 1e-8  | NLopt  | Relative parameter change ($\|\Delta x\| / \|x\| < \text{tol}$).  |
| `xtol_abs` | 1e-10 | NLopt  | Absolute parameter change ($\|\Delta x\| < \text{tol}$).          |

---

## E2 — Machine-Aware & Wrapper Extensions

| Parameter              | Value / Logic | Source      | Role / Description                                        |
| ---------------------- | ------------- | ----------- | --------------------------------------------------------- |
| `finite_diff_rel_step` | auto-scaled   | SciPy       | Relative step for numerical Jacobians if not provided.    |
| `disp`                 | False         | SciPy       | Verbosity flag for convergence messages.                  |
| `workers`              | 1             | SciPy 1.16+ | Number of workers for parallel finite differences.        |

---

# 🔎 Numerical Precisions & Guards

1.  **Epsilon is NOT Unique:**
    *   `eps_machine`: Pure floating-point zero/equality tests.
    *   `eps_rank`: Linear algebra stability ($\sigma < \epsilon_{rank} \implies \text{rank-deficient}$), typically $\sqrt{\epsilon_{machine}}$. Must be separate.

2.  **BFGS Reset (Essential Guard):**
    Reset $B$ to $I$ if: Cholesky fails or curvature conditions repeatedly violated (`max_curvature_fails`). Level R: mandatory fallback.

3.  **NNLS Boundary Ratio Tolerance:**
    $$ \alpha = \min(x_i / (x_i - x_{\text{new}, i})) $$
    Set $\approx 10^{-12}$ to prevent cycling on degenerate constraints.

---

# 🔁 Interaction Matrix (Refined)

| Interaction                        | Effect                     | Notes                                                      |
| ---------------------------------- | -------------------------- | ---------------------------------------------------------- |
| `rho_factor` ↔ `delta_lambda_offset` | Penalty aggressiveness     | Offset ensures dominance; high `rho_factor` risks overshoot. |
| `eps_rank` ↔ `B_reset_tol`           | Hessian fallback frequency | Triggers reset on rank deficiency.                         |
| `w_tol` ↔ `activation_thresh`        | Active-set stability       | Loose `w_tol` may over-activate constraints.               |
| `alpha_ratio_tol` ↔ `w_tol`          | Anti-cycling NNLS          | Ties resolved deterministically.                           |
| `curvature_guard` ↔ `theta_lim`      | BFGS damping frequency     | Guard skips; theta damps for PD preservation.              |
| `eta` ↔ `rho_init`                   | Merit acceptance behavior  | Small `eta` strictens decrease; low `rho` favors objective.|

---

# 🔒 Result

Phase 0.2 is **forensically complete and foundational**.

*   **Status:** Ready for Phase 0.3 integration.

---

# Phase 0.3 – Revision 6 (The Final Specification)

**Status:** Finalized & Audited.
**Referenzen:** Kraft (1988), NLopt `slsqp.c`, Powell (1978), Nocedal & Wright (Numerical Optimization).

---

# 0.3.0 – Constraint Classification Strategy

1.  **Bounds**: $x_{lb} \le x \le x_{ub}$
2.  **Linear**: $A_{lin} x = b_{lin}$
3.  **Nonlinear**: $c(x) = 0$

---

# 0.3.1 – Main SQP Loop

**Logik:** Verwendung der aktualisierten Multiplikatoren $\lambda_{k+1}$ für den neuen Lagrangian-Gradienten. Dies entspricht der klassischen SQP-Theorie (siehe Nocedal & Wright), gewährleistet eine konsistente Approximation der Hesse-Matrix der Lagrangefunktion.

```julia
function slsqp_solve!(ws::SLSQPWorkspace, problem, options)

    initialize_workspace!(ws, problem, options)

    # --- Initial Evaluation ---
    evaluate_at!(ws, ws.x, problem) # Sets f, g, c, jacobian

    # --- Multipliers Initialization ---
    if options.warmstart_lambda !== nothing
        ws.lambda .= options.warmstart_lambda
    else
        fill!(ws.lambda, 0.0)
    end

    # --- Penalty Initialization ---
    # Phase 0.2: Default 10.0 (NLopt/SciPy Production Reality)
    ws.rho = options.rho_init 

    k = 0
    while true
        k += 1

        # --- 1. Convergence Check (Strict Sequential AND) ---
        feas_ok = norm(ws.constraint_violation, Inf) ≤ options.constr_viol_tol
        
        if feas_ok
            # Lagrangian Gradient: ∇L = g + J'λ
            grad_L = ws.g + ws.jacobian' * ws.lambda
            opt_ok = norm(grad_L, Inf) ≤ options.acc
            
            if opt_ok
                return SUCCESS, k
            end
        end

        # --- 2. Termination Limits ---
        if k > ws.max_iter || ws.nfev > ws.max_fun
            return MAXITER_REACHED, k
        end

        # --- 3. QP Subproblem ---
        if ws.total_constraints == 0
            ws.d .= -ws.B \ ws.g
            ws.qp_success = true
        else
            build_qp_subproblem!(ws)
            solve_qp_via_nnls!(ws)
            if !ws.qp_success
                return INFEASIBLE_QP, k
            end
        end

        # --- 4. Line Search ---
        α = line_search!(ws, problem, options)

        if α < ws.alpha_min
            return LINESEARCH_FAIL, k
        end

        # --- 5. State Update & Re-Evaluation ---
        ws.x_new .= ws.x .+ α .* ws.d
        
        # Save OLD Lagrangian Gradient for BFGS
        # ∇L_old = g_old + J_old * λ_old
        ws.grad_L_old .= ws.g .+ ws.jacobian' * ws.lambda

        # Evaluate new point (updates ws.g, ws.jacobian, ws.c)
        evaluate_at!(ws, ws.x_new, problem)

        # Update Multipliers (from QP solution -> λ_new)
        update_multipliers_and_penalty!(ws)

        # --- 6. BFGS Update (On Lagrangian Gradient) ---
        # DESIGN DECISION: We use λ_new for the new gradient.
        # y = ∇L(x_new, λ_new) - ∇L(x_old, λ_old)
        # This is consistent with standard SQP theory.
        ws.grad_L_new .= ws.g .+ ws.jacobian' * ws.lambda
        
        update_hessian!(ws)

        # --- 7. Commit State ---
        ws.x .= ws.x_new
    end
end
```

---

# 0.3.2 – QP Build & Transformation
*(Specification as per Revision 4)*

---

# 0.3.3 – QP → LDP → NNLS
*(Specification as per Revision 4)*

---

# 0.3.4 – Lawson–Hanson NNLS
*(Uses `argmax` for selection, deterministic tie-breaking)*

---

# 0.3.5 – Merit Function & Line Search
*(Uses simplified directional derivative consistent with NLopt)*

---

# 0.3.6 – Multiplier & Penalty Update
*(Standard update logic)*

---

# 0.3.7 – Damped BFGS (On Lagrangian Hessian)

**Korrektur:**
*   Expliziter Reset auf Identitätsmatrix der Dimension $n$.
*   Nutzung von `max_curvature_fails`.

```julia
function update_hessian!(ws)
    
    # Standard SQP: Update on Lagrangian Gradients
    # y = ∇L(x_new, λ_new) - ∇L(x_old, λ_old)
    y = ws.grad_L_new .- ws.grad_L_old
    s = ws.x_new .- ws.x

    sy = dot(s, y)

    # 1. Curvature Guard (Powell)
    if sy ≤ ws.curvature_guard
        ws.curvature_fail_count += 1
        if ws.curvature_fail_count >= ws.max_curvature_fails
            # EXPLICIT RESET to Identity Matrix of dimension n
            # Forensic Note: Ensures dense matrix storage is correctly zeroed.
            fill!(ws.B, 0.0)
            for i in 1:ws.n
                ws.B[i,i] = 1.0
            end
            ws.curvature_fail_count = 0
        end
        return
    end

    ws.curvature_fail_count = 0

    Bs = ws.B * s
    sBs = dot(s, Bs)

    # 2. Powell Damping
    if sy < ws.theta_lim * sBs
        theta = (1.0 - ws.theta_lim) * sBs / (sBs - sy)
        y .= theta .* y .+ (1.0 - theta) .* Bs
        sy = dot(s, y)
    end

    # 3. Rank-2 Update
    ws.B .+= (y*y')/sy - (Bs*Bs')/sBs
end
```

---

# Phase 0.4 – Control-Flow Forensics & Decision Logic

**Ziel:** Definition des Solvers als deterministische Zustandsmaschine.

### 0.4.1 – Die drei Ebenen der Entscheidungslogik

1.  **Hard Exit:** Terminierung (Return to User).
2.  **Soft Recovery:** Interne Zustandsmutation (Lauf wird fortgesetzt).
3.  **Structural Mode Switch:** Wechsel des Algorithmuspfades.

### 0.4.2 – Hard Exit Matrix (Termination)

| Kategorie | Bedingung | Return Code | Semantik |
| :--- | :--- | :--- | :--- |
| **Optimality** | KKT erfüllt & Feasibility ok | `SUCCESS` | Konvergenz erreicht. |
| **Resource Bound** | `k > maxiter` | `MAXITER_REACHED` | User-Limit (Policy). |
| **Resource Bound** | `nfev > maxfun` | `MAXFUN_REACHED` | Wrapper-Policy. |
| **Structural Failure** | QP unlösbar / NNLS Fail | `INFEASIBLE_QP` | Lineares Modell widersprüchlich. |
| **Structural Failure** | `α < alpha_min` | `LINESEARCH_FAIL` | Merit-Function lässt sich nicht senken. |
| **Numerical Error** | `NaN / Inf` in `x, f, g` | `NUMERICAL_ERROR` | Korruption des Zustands. |

### 0.4.3 – Soft Recovery Logic

| Trigger | Aktion | Ort |
| :--- | :--- | :--- |
| `sy ≤ curvature_guard` | Skip BFGS Update | `HessianLayer` |
| `rank(B) < n` | Regularisierung | `HessianLayer` |
| Cholesky Fail (wiederholt) | Reset `B = I` | `HessianLayer` |
| `λmax ≥ rho` | `rho *= rho_factor` | `SLSQPState` |

### 0.4.4 – Determinism Contract

*   Keine zufälligen Pivots.
*   Kein `@fastmath`.
*   Tests mit `BLAS.set_num_threads(1)`.

---

# Phase 0.5 – Architecture Proposal 2.0 (Layered & Forensic)

**Ziel:** Strikte Trennung der Verantwortlichkeiten.

### 0.5.1 – Layered Workspace Design

#### Layer 1: Solver State
```julia
mutable struct SLSQPState{T}
    x::Vector{T}
    x_new::Vector{T}
    g::Vector{T}
    g_new::Vector{T}
    c::Vector{T}
    lambda::Vector{T}
    rho::T
    k::Int
    nfev::Int
end
```

#### Layer 2: Hessian Layer
```julia
mutable struct HessianLayer{T}
    B::Matrix{T}
    regularization_count::Int
end
```

#### Layer 3: QP Engine
```julia
mutable struct QPEngine{T}
    H::Matrix{T}
    A::Matrix{T}
    d::Vector{T}
    lambda_qp::Vector{T}
    nnls::NNLSWorkspace{T}
end
```

#### Layer 4: Orchestrator
```julia
mutable struct SLSQPWorkspace{T}
    state::SLSQPState{T}
    hessian::HessianLayer{T}
    qp::QPEngine{T}
    options::SLSQPOptions{T}
end
```

### 0.5.2 – Architectural Laws

1.  **Isolation:** Kein Modul kennt höhere Layer.
2.  **Memory Transparency:** Datenflüsse sind explizit.
3.  **Type Stability:** Alle Felder konkret typisiert.

---

# Phase 0.6 – Go/No-Go & Closure

### 0.6.1 – Reproduction Readiness Checklist

| Kriterium | Status |
| :--- | :--- |
| Konstanten fixiert (0.2) | ✅ |
| Constraint-Trennung (0.3) | ✅ |
| Exit Semantik (0.4) | ✅ |
| Architektur Modular (0.5) | ✅ |
| Determinism Contract | ✅ |

### 0.6.2 – Decision

**Entscheid: GO for Phase 1.**

Die Spezifikation ist wasserdicht. Risiken sind bekannt und tolerierbar.
