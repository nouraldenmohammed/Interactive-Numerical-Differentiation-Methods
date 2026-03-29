import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy

# --------------------------------------------------------
# NUMERICAL DIFFERENTIATION METHODS
# --------------------------------------------------------
def forward_diff(f, x, h):
    return (f(x + h) - f(x)) / h

def backward_diff(f, x, h):
    return (f(x) - f(x - h)) / h

def central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

def central_diff_2nd(f, x, h):
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h**2)

def richardson_table(f, x, h, n):
    F = np.zeros((n, n))
    for i in range(n):
        hi = h / (2**i)
        F[i, 0] = central_diff(f, x, hi)

    for j in range(1, n):
        for i in range(j, n):
            multiplier = 4**j
            F[i, j] = (multiplier * F[i, j-1] - F[i-1, j-1]) / (multiplier - 1)
            
    return F

# --------------------------------------------------------
# --------------------------------------------------------
# HELPER FOR MATH EVALUATION (SYMPY)
# --------------------------------------------------------
@st.cache_resource
def get_derivatives(func_str, var_str='x'):
    """Parses a string into a callable function and its sympy derivatives."""
    try:
        # Clean up common numpy prefixes to allow native sympy parsing
        func_str = func_str.replace("np.", "")
        x_sym = sympy.symbols(var_str)
        func_sym = sympy.sympify(func_str)

        # Lambdify for numerical evaluation
        f = sympy.lambdify(x_sym, func_sym, 'numpy')

        # First derivative
        df_sym = sympy.diff(func_sym, x_sym)
        df_exact = sympy.lambdify(x_sym, df_sym, 'numpy')

        # Second derivative
        d2f_sym = sympy.diff(df_sym, x_sym)
        d2f_exact = sympy.lambdify(x_sym, d2f_sym, 'numpy')

        return f, df_exact, d2f_exact, func_sym, df_sym, d2f_sym, None
    except Exception as e:
        return None, None, None, None, None, None, str(e)

# --------------------------------------------------------
# STREAMLIT APP LAYOUT
# --------------------------------------------------------
st.set_page_config(page_title="Numerical Differentiation Applet", layout="wide")
st.title("Interactive Numerical Differentiation")
st.markdown("*By Dr Nouralden Mohammed*")

tab1, tab2, tab3 = st.tabs(["1D Derivative Rules", "Richardson Extrapolation", "Second Derivatives"])

# --- TAB 1: First Derivatives & Error ---
with tab1:
    st.header("Standard 1D Derivative Methods")
    st.markdown("Compare Forward, Backward, and Central Difference formulas for $f'(x)$.")
    
    col1, col2 = st.columns([1, 2.5])
    with col1:
        st.markdown("**Function Definition:**")
        func_expr = st.text_input("f(x):", value="exp(x)")
        
        st.markdown("**Parameters:**")
        c1, c2 = st.columns(2)
        with c1:
            x_val = st.number_input("Evaluate at (x)", value=1.0)
        with c2:
            h_val = st.number_input("Step size (h)", value=0.1, step=0.01, format="%.5f")
            
        methods_selected = st.multiselect(
            "Select Methods to Compare:",
            ["Forward", "Backward", "Central"],
            default=["Forward", "Central"]
        )

    with col2:
        f, df_exact, d2f_exact, func_sym, df_sym, d2f_sym, err = get_derivatives(func_expr)
        if err:
            st.error(f"Error evaluating mathematical expression: {err}")
        else:
            exact_val = df_exact(x_val)
            st.markdown(f"**Symbolic Derivative:** $f'(x) = {sympy.latex(df_sym)}$")
            st.markdown(f"**Exact f'({x_val}) Value:** `{exact_val:.8f}`")
            
            # Plotting Setup
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            h_steps = np.logspace(-16, -1, 100)
            
            results_data = []
            
            # Calculate & Plot based on selections
            if "Forward" in methods_selected:
                res_fd = forward_diff(f, x_val, h_val)
                err_fd = abs(exact_val - res_fd)
                results_data.append({"Method": "Forward Diff.", "Result": res_fd, "Error": err_fd})
                errors_fd = [abs(exact_val - forward_diff(f, x_val, h)) for h in h_steps]
                ax1.loglog(h_steps, errors_fd, label="Forward Difference O(h)", color='blue')
                
            if "Backward" in methods_selected:
                res_bd = backward_diff(f, x_val, h_val)
                err_bd = abs(exact_val - res_bd)
                results_data.append({"Method": "Backward Diff.", "Result": res_bd, "Error": err_bd})
                errors_bd = [abs(exact_val - backward_diff(f, x_val, h)) for h in h_steps]
                ax1.loglog(h_steps, errors_bd, label="Backward Difference O(h)", color='orange')
                
            if "Central" in methods_selected:
                res_cd = central_diff(f, x_val, h_val)
                err_cd = abs(exact_val - res_cd)
                results_data.append({"Method": "Central Diff.", "Result": res_cd, "Error": err_cd})
                errors_cd = [abs(exact_val - central_diff(f, x_val, h)) for h in h_steps]
                ax1.loglog(h_steps, errors_cd, label="Central Difference O(h^2)", color='green')

            ax1.set_xlabel('Step size (h)')
            ax1.set_ylabel('Absolute Error')
            ax1.set_title(f"Error Analysis: Truncation vs Round-off at x = {x_val}")
            ax1.axvline(1e-7, color='r', linestyle='--', label='Theoretical Optimal h (sqrt(eps))')
            ax1.legend()
            ax1.grid(True, which="both", ls=":", alpha=0.6)
            ax1.set_ylim(bottom=1e-15)
            st.pyplot(fig1)
            
            # Display results
            if results_data:
                st.table(pd.DataFrame(results_data).set_index("Method").style.format({"Result": "{:.6f}", "Error": "{:.2e}"}))

# --- TAB 2: Richardson Extrapolation ---
with tab2:
    st.header("Richardson Extrapolation")
    st.markdown("Generates a table of approximations using Richardson Extrapolation to eliminate error terms.")
    
    col3, col4 = st.columns([1, 2.5])
    with col3:
        st.markdown("**Function Definition:**")
        func_expr_romb = st.text_input("f(x) for Extrapolation:", value="x**2 * cos(x)", key="romb_f")
        
        st.markdown("**Parameters:**")
        c3, c4 = st.columns(2)
        with c3:
            x_romb = st.number_input("Evaluate at (x)", value=1.0, key="romb_x")
        with c4:
            h_romb = st.number_input("Initial Step Size (h)", value=0.1, key="romb_h")
            
        n_rows = st.slider("Number of Rows (n)", 2, 8, 3)

    with col4:
        f2, df_exact2, _, func_sym2, df_sym2, _, err2 = get_derivatives(func_expr_romb)
        if err2:
            st.error(f"Error evaluating mathematical expression: {err2}")
        else:
            exact_romb = df_exact2(x_romb)
            st.markdown(f"**Symbolic Derivative:** $f'(x) = {sympy.latex(df_sym2)}$")
            st.markdown(f"**Exact f'({x_romb}) Value:** `{exact_romb:.8f}`")
            
            # Compute Richardson Table
            R = richardson_table(f2, x_romb, h_romb, n_rows)
            
            # Format table nicely
            columns = [f"O(h^{2*(j+1)})" for j in range(n_rows)]
            index = [f"i={i} (h={h_romb / 2**i})" for i in range(n_rows)]
            
            df_R = pd.DataFrame(R, columns=columns, index=index)
            df_R = df_R.applymap(lambda x: f"{x:.8f}" if x != 0.0 else "")
            
            st.markdown("**Richardson Table:**")
            st.dataframe(df_R, use_container_width=True)
            
            best_est = R[-1, -1]
            st.success(f"**Best Estimate:** `{best_est:.8f}`  |  **Absolute Error:** `{abs(exact_romb - best_est):.2e}`")

# --- TAB 3: Second Derivatives ---
with tab3:
    st.header("Second Derivative Approximation")
    st.markdown("Approximates the second derivative $f''(x)$ using the central difference formula.")
    
    col5, col6 = st.columns([1, 2.5])
    with col5:
        st.markdown("**Function Definition:**")
        func_2d_expr = st.text_input("f(x):", value="exp(x)", key="f_tab3")
        
        st.markdown("**Parameters:**")
        c5, c6 = st.columns(2)
        with c5:
            x_val_3 = st.number_input("Evaluate at (x)", value=1.0, key="x_tab3")
        with c6:
            h_val_3 = st.number_input("Step size (h)", value=0.1, step=0.01, format="%.5f", key="h_tab3")

    with col6:
        f3, _, d2f_exact3, func_sym3, _, d2f_sym3, err3 = get_derivatives(func_2d_expr)
        if err3:
            st.error(f"Error evaluating mathematical expression: {err3}")
        else:
            exact_val_f2 = d2f_exact3(x_val_3)
            res_2nd = central_diff_2nd(f3, x_val_3, h_val_3)
            err_2nd = abs(exact_val_f2 - res_2nd)
            
            st.metric(label="Approximate f''(x)", value=f"{res_2nd:.6f}", delta=f"Error: {err_2nd:.2e}", delta_color="inverse")
            st.markdown(f"**Symbolic 2nd Derivative:** $f''(x) = {sympy.latex(d2f_sym3)}$")
            st.markdown(f"**Exact f''({x_val_3}) Value:** `{exact_val_f2:.6f}`")
            
            # Error plot
            h_steps_2 = np.logspace(-8, -1, 100)
            errors_cd_2 = [abs(exact_val_f2 - central_diff_2nd(f3, x_val_3, h)) for h in h_steps_2]

            fig3, ax3 = plt.subplots(figsize=(10, 5))
            ax3.loglog(h_steps_2, errors_cd_2, label="Central Difference O(h^2)", color="purple")
            ax3.set_xlabel("Step size (h)")
            ax3.set_ylabel("Absolute Error")
            ax3.set_title(f"Error Analysis for f''(x) at x = {x_val_3}")
            ax3.axvline(1e-5, color='r', linestyle='--', label='Theoretical Optimal h (cbrt(eps))')
            ax3.legend()
            ax3.grid(True, which="both", ls=":", alpha=0.6)
            ax3.set_ylim(bottom=1e-13)
            st.pyplot(fig3)
