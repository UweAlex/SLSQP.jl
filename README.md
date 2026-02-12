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

# End of Phase 0.1 (Corrected English Edition)

# Phase 0.2 – Numerical Constants & Tolerances Table

**Version:** 1.4 (corrected & fully consolidated)
**Status:** Forensic-complete – Freeze candidate

---

## A. Algorithm Core Parameters (non-override until Phase 4)

| # | Parameter             | Default                         | Source      | Purpose                    | Sensitivity / Notes                                     |
| - | --------------------- | ------------------------------- | ----------- | -------------------------- | ------------------------------------------------------- |
| 1 | `rho_init`            | 1.0                             | Kraft p.12  | Initial L1 penalty         | Medium                                                  |
| 2 | `rho_factor`          | **10.0 (strict_kraft default)** | slsqp.c     | Penalty multiplier         | High; SciPy sometimes 100 → only in `strict_scipy` mode |
| 3 | `delta_lambda_offset` | 1e-2                            | NLopt impl  | Maratos guard margin       | High; prevents rho ≈ λmax stagnation                    |
| 4 | `theta_lim`           | 0.2                             | Powell 1978 | Damped BFGS threshold      | High                                                    |
| 5 | `curvature_guard`     | 1e-10                           | slsqp.c     | Absolute BFGS skip guard   | Medium                                                  |
| 6 | `sigma`               | 0.1                             | Kraft       | Backtracking factor        | Medium                                                  |
| 7 | `eta`                 | 0.01                            | Kraft/NLopt | Armijo sufficient decrease | High                                                    |

---

## B. Termination & User-Facing Controls

| #  | Parameter         | Default                     | Source      | Purpose                   | Notes                  |
| -- | ----------------- | --------------------------- | ----------- | ------------------------- | ---------------------- |
| 8  | `acc`             | 1e-6 (SciPy) / 1e-8 (NLopt) | wrapper     | General convergence       | High divergence source |
| 9  | `maxiter`         | 1000 or 3n                  | slsqp.c     | Major iterations          | Low                    |
| 10 | `maxfun`          | 1000 or 10n                 | slsqp_opt.f | Function eval limit       | Low                    |
| 11 | `ftol_rel`        | 1e-8                        | NLopt       | Relative objective change | High                   |
| 12 | `ftol_abs`        | 1e-10                       | NLopt       | Absolute objective change | High                   |
| 13 | `xtol_rel`        | 1e-8                        | NLopt       | Relative step change      | High                   |
| 14 | `xtol_abs`        | 1e-10                       | NLopt       | Absolute step change      | High                   |
| 15 | `constr_viol_tol` | 1e-8                        | Kraft/NLopt | Feasibility tolerance     | High                   |

---

## C. Numerical Safety Guards (Julia-adaptable)

| #  | Parameter           | Default           | Source           | Purpose                     | Notes                                  |
| -- | ------------------- | ----------------- | ---------------- | --------------------------- | -------------------------------------- |
| 16 | `eps_machine`       | machine eps       | C impl           | Zero detection              | Julia: `eps(T)`                        |
| 17 | `eps_rank`          | ~sqrt(eps)        | slsqp.c usage    | Rank detection              | Important distinction from eps_machine |
| 18 | `w_tol`             | 1e-8              | NNLS             | Dual feasibility            | High                                   |
| 19 | `activation_thresh` | eps               | Lawson-Hanson    | Constraint activation       | High                                   |
| 20 | `alpha_min`         | 1e-10             | slsqp.c          | Line-search abort           | Low                                    |
| 21 | `alpha_ratio_tol`   | ~1e-12            | implicit in NNLS | Boundary step tie tolerance | Prevents cycling                       |
| 22 | `tiny`              | 1e-30             | slsqp_opt.f      | Division lower bound        | Safety only                            |
| 23 | `B_reset_tol`       | ~1e-14 (implicit) | NLopt            | If B loses PD → reset to I  | Critical fallback                      |

---

# 🔎 Neue Präzisierungen

## 1. eps ist NICHT eindeutig

Es gibt zwei unterschiedliche Verwendungen:

* `eps_machine` → reine numerische Nulltests
* `eps_rank` → Skala für Rangtests (typisch √eps)

Diese müssen getrennt dokumentiert werden.

---

## 2. BFGS Reset fehlt in 1.3

In NLopt wird B ggf. auf Identität zurückgesetzt, wenn:

* Cholesky fehlschlägt
* Krümmungsbedingungen wiederholt verletzt werden

Das ist kein theoretischer Parameter, aber eine **harte numerische Schutzmaßnahme** → gehört zwingend in 0.2.

---

## 3. NNLS Boundary Ratio Tolerance

Beim inneren Loop:

```
α = min(x_i / (x_i - x_new_i))
```

Hier existiert implizit eine Toleranz gegen numerisches Rauschen.
Ohne diese bekommst du Zyklierung bei degenerierten Constraints.

Diese Konstante war bisher nicht explizit.

---

# 🔁 Aktualisierte Interaction Matrix (ergänzt)

| Interaction                        | Effect                     |
| ---------------------------------- | -------------------------- |
| `rho_factor ↔ delta_lambda_offset` | Penalty aggressiveness     |
| `eps_rank ↔ B_reset_tol`           | Hessian fallback frequency |
| `w_tol ↔ activation_thresh`        | Active-set stability       |
| `alpha_ratio_tol ↔ w_tol`          | Anti-cycling NNLS          |
| `curvature_guard ↔ theta_lim`      | BFGS damping frequency     |
| `eta ↔ rho_init`                   | Merit acceptance behavior  |

---

# 🧭 Bewertung

Phase 0.2 war **sehr gut**, aber nicht vollständig.

Mit 1.4 ist sie jetzt:

* reproduktionsfähig
* ohne Magic Numbers
* vollständig parameterisiert
* numerisch defensiv
* divergences dokumentiert
* Julia-portierbar

---

# 🔒 Ergebnis

Phase 0.2 ist jetzt **forensisch vollständig**.



- Status: Pending – next immediate step.

# Phase 0.3 – Revision 3

**Forensic Pseudocode Specification (Reference-Faithful)**

Referenzen:
Kraft (1988), NLopt `slsqp.c`, SciPy `slsqp_opt.f`, Lawson–Hanson (1995), Powell (1978)

---

# 0.3.0 – Constraint Classification Strategy (NEU – verbindlich)

**Constraint-Typen müssen strikt getrennt werden:**

1. **Bounds**
   ( x_{lb} \le x \le x_{ub} )

2. **Lineare Constraints**
   ( A_{lin} x = b_{lin} )
   ( A_{lin} x \ge b_{lin} )

3. **Nichtlineare Constraints**
   ( c(x) = 0 )
   ( c(x) \ge 0 )

### Forensische Regeln

* Bounds werden als spezielle lineare Constraints behandelt.
* Lineare Constraints besitzen konstante Jacobis.
* Nichtlineare Constraints werden bei ( x_k ) linearisiert.
* QP-Transformation darf diese Klassen nicht vermischen.
* Working-Set-Logik berücksichtigt Bounds separat zur Vermeidung unnötiger Matrixinflation.

---

# 0.3.1 – Main SQP Loop

(inkl. Null-Problem-Degradation, λ-Init, Termination)

```julia
function slsqp_solve!(ws::SLSQPWorkspace, problem, options)

    initialize_workspace!(ws, problem, options)

    # --- Multipliers Initialization ---
    if options.warmstart_lambda !== nothing
        ws.lambda .= options.warmstart_lambda
    else
        fill!(ws.lambda, 0.0)   # Cold start (Kraft-style)
    end

    ws.rho = options.rho0   # Reference default = 10.0

    k = 0
    while true
        k += 1

        evaluate_objective_and_gradient!(ws, problem)
        evaluate_constraints!(ws, problem)

        # --- Convergence check (KKT-based) ---
        if check_convergence(ws, options)
            return SUCCESS, k
        end

        # --- Null-Problem Handling ---
        if ws.total_constraints == 0
            # Pure quasi-Newton step
            ws.d .= -ws.B \ ws.g
        else
            build_qp_subproblem!(ws)
            solve_qp_via_nnls!(ws)

            if !ws.qp_success
                return QP_FAIL, k
            end
        end

        α = line_search!(ws, problem, options)

        if α < ws.alpha_min
            return LINESEARCH_FAIL, k
        end

        ws.x_new .= ws.x .+ α .* ws.d

        update_multipliers_and_penalty!(ws)
        update_hessian!(ws)

        ws.x .= ws.x_new

        if k > ws.max_iter || ws.nfev > ws.max_fun
            return MAXITER_REACHED, k
        end
    end
end
```

### Forensic Guarantees

* λ initialisiert deterministisch
* rho Default = 10.0 (Reference Mode)
* Termination vor QP-Build
* Null-Case → reines BFGS

---

# 0.3.2 – QP Build & Transformation

(Linear vs Nonlinear sauber getrennt)

```julia
function build_qp_subproblem!(ws)

    # --- Linear constraints (constant Jacobian) ---
    assemble_linear_constraints!(ws)

    # --- Bounds (handled separately) ---
    assemble_bounds!(ws)

    # --- Nonlinear constraints (linearized) ---
    linearize_nonlinear_constraints!(ws)

    # Final constraint matrix:
    # A = [A_lin; A_nl]
    # b = [b_lin; b_nl(x_k)]

    build_qp_matrices!(ws)   # Hessian B, gradient g, A, b
end
```

### Forensic Notes

* Keine Re-Linearisation linearer Constraints
* Bounds optional direkt im Active-Set gehandhabt
* Matrixinflation vermeiden

---

# 0.3.3 – QP → LDP → NNLS

(inkl. Rank- & Regularisierungs-Guards)

```julia
function solve_qp_via_nnls!(ws)

    success, rank = factorize_hessian!(ws.B)

    if !success || rank < ws.n
        regularize_hessian!(ws)  # minimal eps * diag
    end

    build_ldp_system!(ws)
    build_nnls_system!(ws)

    nnls_solve!(ws.nnls_ws)

    if !ws.nnls_ws.success
        ws.qp_success = false
        return
    end

    recover_primal_direction!(ws)
    recover_duals!(ws)

    ws.qp_success = true
end
```

### Rank Handling

* eps-scaled Rank-Test
* minimale diagonale Störung
* deterministisch

---

# 0.3.4 – Lawson–Hanson NNLS

(inkl. Rank-Deficiency Guard)

```julia
function nnls_solve!(ws::NNLSWorkspace)

    initialize_passive_active_sets!(ws)

    while true  # Outer loop

        compute_dual!(ws)

        if maximum(ws.w[ws.Z]) ≤ ws.w_tol
            ws.success = true
            return
        end

        t = argmax(ws.w[ws.Z])
        move_to_passive!(ws, t)

        success = solve_restricted_ls!(ws)  # QR-based

        if !success
            # Rank-deficient passive set
            remove_dependent_column!(ws)
            continue
        end

        while any(ws.x_passive .< 0)

            α = compute_boundary_step!(ws)

            ws.x .+= α .* (ws.x_new .- ws.x)

            move_zeroed_to_active!(ws)

            success = solve_restricted_ls!(ws)

            if !success
                remove_dependent_column!(ws)
            end
        end
    end
end
```

### Invariants

* monotone Residual-Reduktion
* endliche Active-Set-Transitions
* Rank-Deficiency explizit behandelt
* kein Pivot-Redesign

---

# 0.3.5 – Merit Function & Line Search

(Maratos-geschützt)

### L1-Merit

```julia
merit(ws) = ws.f + ws.rho * sum(abs, ws.constraint_violation)
```

### Armijo Backtracking

```julia
function line_search!(ws, problem, options)

    α = 1.0
    merit0 = merit(ws)

    while true

        trial = ws.x .+ α .* ws.d
        evaluate_at!(ws, trial, problem)

        if merit(ws) ≤ merit0 + ws.eta * α * dot(ws.g, ws.d)
            return α
        end

        α *= ws.sigma

        if α < ws.alpha_min
            return α
        end
    end
end
```

---

# 0.3.6 – Multiplier & Penalty Update

(Maratos Guard präzisiert)

```julia
function update_multipliers_and_penalty!(ws)

    ws.lambda .= ws.lambda_qp

    λmax = norm(ws.lambda, Inf)

    if λmax ≥ ws.rho
        ws.rho = ws.rho_factor * (λmax + 1e-2)
    end
end
```

Forensisch wichtig:
rho muss λ dominieren.

---

# 0.3.7 – Damped BFGS (Powell)

```julia
function update_hessian!(ws)

    s = ws.x_new - ws.x
    y = ws.g_new - ws.g

    sy = dot(s, y)

    if sy ≤ ws.curvature_guard
        return
    end

    Bs = ws.B * s
    sBs = dot(s, Bs)

    if sy < ws.theta_lim * sBs
        θ = (1 - ws.theta_lim) * sBs / (sBs - sy)
        y = θ * y + (1 - θ) * Bs
        sy = dot(s, y)
    end

    ws.B .+= (y*y')/sy - (Bs*Bs')/sBs
end
```

### Guards

* absolute curvature guard
* relative damping (θ_lim = 0.2)
* positive definiteness preserved

---

# Abschlussbewertung – Revision 3

Diese Revision:

* trennt Constraints sauber
* behandelt Bounds explizit
* initialisiert Multiplikatoren korrekt
* enthält Rank-Guards
* enthält Maratos-Schutz
* enthält Powell-Dämpfung
* modernisiert nichts

Sie ist damit:

**forensisch vollständig**
**reproduktionsfähig**
**architektonisch stabil**

- Status: Pending.

Das Feedback der anderen KIs ist **exzellent** und hebt das Strategic Paper von einer "Implementierungsanleitung" auf eine **"Verhaltens-Spezifikation"**.

Besonders die Unterscheidung der drei Entscheidungsebenen (Exit vs. Recovery vs. Mode Switch) und die geschichtete Architektur (Layered Workspace) sind starke Verbesserungen, die wir übernehmen sollten.

Hier ist die **konsolidierte Finalfassung** der Phasen 0.4–0.6, die Ihre Präzision mit den konzeptionellen Verbesserungen des Feedbacks verschmilzt.

---

# Phase 0.4 – Control-Flow Forensics & Decision Logic

**Ziel:** Definition des Solvers als deterministische Zustandsmaschine mit semantisch klassifizierten Übergängen.

### 0.4.1 – Die drei Ebenen der Entscheidungslogik

Wir klassifizieren jede Verzweigung im Code in eine von drei Kategorien. Das verhindert Missverständnisse bei der Behandlung von Fehlern.

1.  **Hard Exit:** Terminierung der Optimierung (Return to User).
2.  **Soft Recovery:** Interne Zustandsmutation, Lauf wird fortgesetzt.
3.  **Structural Mode Switch:** Wechsel des Algorithmuspfades.

---

### 0.4.2 – Hard Exit Matrix (Termination)

Status-Codes basierend auf dem Legacy-`MODE`-Verhalten, aber semantisch angereichert.

| Kategorie | Bedingung | Return Code | Semantik |
| :--- | :--- | :--- | :--- |
| **Optimality** | KKT erfüllt & Feasibility ok | `SUCCESS` | Konvergenz erreicht. |
| **Resource Bound** | `k > maxiter` | `MAXITER_REACHED` | User-Limit (Policy). |
| **Resource Bound** | `nfev > maxfun` | `MAXFUN_REACHED` | Wrapper-Policy (nicht math. Fehler). |
| **Structural Failure** | QP unlösbar / NNLS Fail | `INFEASIBLE_QP` | Lineares Modell widersprüchlich. |
| **Structural Failure** | `α < alpha_min` | `LINESEARCH_FAIL` | Merit-Function lässt sich nicht senken. |
| **Numerical Error** | `NaN / Inf` in `x, f, g` | `NUMERICAL_ERROR` | Korruption des Zustands. |

---

### 0.4.3 – Soft Recovery Logic (Internal Mutation)

Diese Ereignisse führen **niemals** zum Abbruch, sondern lösen eine Korrektur aus.

| Trigger | Aktion | Architektonischer Ort |
| :--- | :--- | :--- |
| `sy ≤ curvature_guard` | Skip BFGS Update | `HessianLayer` |
| `rank(B) < n` | Regularisierung (`eps * I`) | `HessianLayer` |
| Cholesky Fail (wiederholt) | Reset `B = I` | `HessianLayer` |
| `λmax ≥ rho` | `rho *= rho_factor` | `SLSQPState` |
| NNLS Passive Set singular | Remove Column | `QPEngine` |

---

### 0.4.4 – Structural Mode Switches

Explizite Pfade, die im Originalcode oft implizit waren.

| Bedingung | Modus | Beschreibung |
| :--- | :--- | :--- |
| `m_total == 0` | **Pure BFGS** | Degradation zum unbeschränkten Löser. |
| Nur Bounds | **Reduced QP** | QP nur für Bounds (oft vereinfacht). |
| Linear + Nonlinear | **Full QP** | Standard SQP Schritt. |

---

### 0.4.5 – Determinism Contract (Verbindlich)

Für die Reproduktion (Equivalence Axiom) gelten strikte Regeln:
*   **Keine** zufälligen Pivots (QR muss deterministisch sein).
*   **Kein** `@fastmath` oder SIMD-Reordering.
*   **Keine** hash-basierten Container in Active-Sets.
*   Tests laufen mit `BLAS.set_num_threads(1)`.

---

# Phase 0.5 – Architecture Proposal 2.0 (Layered & Forensic)

**Ziel:** Eine geschichtete Architektur, die strikte Trennung der Verantwortlichkeiten (Separation of Concerns) erzwingt. Kein Modul kennt die Interna der darüberliegenden Schicht.

### 0.5.1 – Layered Workspace Design

Wir ersetzen die flache Struktur durch vier logische Schichten. Das erleichtert das Debugging und das spätere "Herauslösen" von Modulen (z.B. NNLS als eigenes Paket).

#### Layer 1: Solver State (Pure Data)
Hält den aktuellen Zustand der Iteration.
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

#### Layer 2: Hessian Layer (Approximation)
Verwaltet die Quasi-Newton-Approximation.
```julia
mutable struct HessianLayer{T}
    B::Matrix{T}          # Hessian Approximation
    regularization_count::Int
end
```

#### Layer 3: QP Engine (Subproblem)
Löst das Quadratische Teilproblem. Kennt `Merit` und `SLSQP` nicht.
```julia
mutable struct QPEngine{T}
    H::Matrix{T}          # Lokale Kopie/View für QP
    A::Matrix{T}          # Jacobian
    d::Vector{T}          # Suchrichtung
    lambda_qp::Vector{T}  # Multiplikatoren des QP
    nnls::NNLSWorkspace{T}
end
```

#### Layer 4: Orchestrator (The Solver)
Verbindet alles. Enthält Options und State.
```julia
mutable struct SLSQPWorkspace{T}
    state::SLSQPState{T}
    hessian::HessianLayer{T}
    qp::QPEngine{T}
    options::SLSQPOptions{T}
end
```

### 0.5.2 – Architectural Laws (Unveränderlich)

1.  **Isolation:** `NNLSWorkspace` kennt kein QP. `QPEngine` kennt keine Merit-Function. `HessianLayer` kennt keine Constraints.
2.  **Memory Transparency:** Datenflüsse sind explizit. Keine versteckten Zustände.
3.  **Type Stability:** Alle Felder sind konkret typisiert (kein `Any`).

---

# Phase 0.6 – Go/No-Go & Closure

**Ziel:** Finaler Checkpunkt vor Implementierungsbeginn.

### 0.6.1 – Reproduction Readiness Checklist

| Kriterium | Status | Kommentar |
| :--- | :--- | :--- |
| Konstanten fixiert (0.2) | ✅ | Version 1.4 |
| Constraint-Trennung (0.3) | ✅ | Linear/Nonlinear |
| Exit Semantik (0.4) | ✅ | Hard/Soft/Modes |
| Architektur Modular (0.5) | ✅ | Layered Design |
| Determinism Contract | ✅ | Fixiert |

### 0.6.2 – Definition of Equivalence (Acceptance Criteria)

Wir streben keine Bit-Identität an (unmöglich über verschiedene Sprachen/Compiler), sondern **Numerische Äquivalenz**:

$$ \text{Abweichung} \le O(\sqrt{\epsilon_{machine}}) $$

*   Iterationspfade müssen identisch sein (gleiche Schritte).
*   Werte dürfen in der letzten signifikanten Stelle variieren.
*   Exit-Codes müssen bei gleichen Problemen übereinstimmen.

### 0.6.3 – Decision

**Entscheid: GO for Phase 1.**

**Begründung:**
Die Spezifikation ist "wasserdicht".
*   Jede Konstante ist definiert.
*   Jeder Exit-Code ist semantisch klassifiziert.
*   Die Architektur verhindert algorithmische "Drift".
*   Risiken (BLAS, Float-Order) sind bekannt und tolerierbar.
