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



