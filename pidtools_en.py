# python3

"""
Reference discrete-PID implementation, included so that the Experiment 2
baselines (Section 5.2) are self-contained. The two-source and N-source
synergy / redundancy / unique-information formulas follow the
"conditional-copy decomposition" framework of a companion paper that is
likewise under review; only the TSE and N-order-synergy entry points
(`total_syn_effect`, `multi_source_syn`) are exercised by
experiment2_run.py.

Logarithms are base 2 (bits). All other modules in this repository
default to nats.

All variable / column names in the example correspond to the original
string positions inside each outcome, i.e. the first bit -> 'X', the
second -> 'Y', and the third -> 'Z'.
"""

from __future__ import annotations
import itertools
from typing import List
import pandas as pd
import numpy as np

DataFrame = pd.DataFrame

# ---------------------------------------------------------------------------
# Convert a `Distribution` object into a joint-distribution DataFrame.
# ---------------------------------------------------------------------------
def distribution_to_dataframe(dist, var_names: List[str]) -> DataFrame:
    rows = []
    for outcome, p in zip(dist.outcomes, dist.probabilities):
        if len(outcome) != len(var_names):
            raise ValueError("Length mismatch between outcome and var_names")
        row = list(outcome) + [p]
        rows.append(row)

    columns = var_names + ["Pr"]
    df = pd.DataFrame(rows, columns=columns)
    df["Pr"] = df["Pr"] / df["Pr"].sum()
    return df

# ---------------------------------------------------------------------------
# Extract marginal distributions and, given the combination order, treat
# every order-sized subset of sources as a new combined variable; return
# a new joint-distribution DataFrame.
# ---------------------------------------------------------------------------
def df_to_new_df(df: DataFrame, src: List[str], tgt: str, order: int) -> DataFrame:
    """
    Take the marginal distribution over (src, tgt), then for every
    size-`order` subset of src treat it as a single combined variable
    whose value is the comma-joined string of its components, and return
    the new joint-distribution DataFrame.

    Parameters
    ----------
    df    : DataFrame containing at least the src columns, the tgt column
            and a 'Pr' column.
    src   : List of source variable names.
    tgt   : Target variable column name.
    order : Subset size, must lie in [1, len(src)-1].

    Returns
    -------
    ndf   : New joint-distribution DataFrame with column order
            ['Pr'] + all '(Si,Sj,...)' combination columns + [tgt].

    Raises
    ------
    ValueError if `order` is not in [1, len(src)-1].
    """
    # Validate order.
    if order < 1 or order > len(src):
        raise ValueError(f"order ({order}) must lie between 1 and len(src) ({len(src)})")
    # 1) Marginalise over (src + tgt) by groupby + sum.
    df_m = df.groupby(src + [tgt], sort=False)["Pr"].sum().reset_index()

    # 2) Enumerate size-`order` subsets of src.
    combos = list(itertools.combinations(src, order))
    # 3) Cast every src-column value to an integer-string for clean joining.
    for v in src:
        df_m[v] = df_m[v].astype(str).astype(str)
    # 4) For every subset, build a new column named '(S1,S2)' whose value
    #    is e.g. '0,1'.
    for subset in combos:
        col_name = f"({','.join(subset)})"
        df_m[col_name] = df_m.apply(
            lambda row: ",".join(row[v] for v in subset),
            axis=1
        )
    # 5) Set the new column order: 'Pr' first, combination columns next,
    #    target last.
    new_cols = ["Pr"] + [f"({','.join(sub)})" for sub in combos] + [tgt]
    ndf = df_m[new_cols].copy()
    # 6) Print a sample for debugging.
    # print("sample of generated ndf:")
    # print(ndf.head())
    return ndf

# ---------------------------------------------------------------------------
# Test helper: random joint distribution.
# ---------------------------------------------------------------------------
def generate_random_df(var_names: List[str]) -> DataFrame:
    """
    Random binary joint-distribution DataFrame for the given variable list.
    Each variable takes values 0 or 1, with random (normalised) probabilities.

    Parameters
    ----------
    var_names: list of all variables (sources and target).

    Returns
    -------
    rand_df : the generated joint-distribution DataFrame, with columns
              var_names + ['Pr'].
    """
    # Enumerate all binary combinations.
    outcomes = list(itertools.product([0, 1, 2], repeat=len(var_names)))
    probs = np.random.rand(len(outcomes))
    probs /= probs.sum()

    rows = []
    for outcome, p in zip(outcomes, probs):
        rows.append(list(outcome) + [p])
    columns = var_names + ["Pr"]
    rand_df = pd.DataFrame(rows, columns=columns)
    return rand_df


# ---------------------------------------------------------------------------
# Build the conditionally-independent surrogate distribution
# Q(x', y', ... | target).
# ---------------------------------------------------------------------------
def build_conditionally_independent_df(df: DataFrame, target_vars: List[str]) -> DataFrame:
    """
    From the input DataFrame, build a new joint distribution Q under which
    every non-target variable is conditionally independent given target_vars.
    Column names are kept identical to the input (no primes appended).

    Steps:
      1) Compute the target marginal P(target_vars).
      2) Compute the conditional P(v | target) for every non-target variable v.
      3) Enumerate all combinations and form Q = P(target) * prod_v P(v | target).
      4) Output column order: ['Pr'] + other_vars + target_vars.
    """
    if "Pr" not in df.columns:
        raise KeyError("input DataFrame must contain a 'Pr' column")
    # Identify non-target variables.
    other_vars = [c for c in df.columns if c not in target_vars + ["Pr"]]
    # 1) Target marginal P(target_vars).
    p_target = df.groupby(target_vars, sort=False)["Pr"].sum()
    # 2) Conditional probability P(v | target) for every non-target variable.
    cond_probs: dict = {}
    for tgt_vals, group in df.groupby(target_vars, sort=False):
        key = tgt_vals if isinstance(tgt_vals, tuple) else (tgt_vals,)
        cond_probs[key] = {}
        total = group["Pr"].sum()
        for var in other_vars:
            cond_probs[key][var] = group.groupby(var, sort=False)["Pr"].sum() / total
    # 3) Enumerate all combinations and rebuild the joint Q.
    new_rows = []
    for tgt_vals, p_t in p_target.items():
        key = tgt_vals if isinstance(tgt_vals, tuple) else (tgt_vals,)
        values_list = [cond_probs[key][v].index.tolist() for v in other_vars]
        for combo in itertools.product(*values_list):
            prob = p_t
            for v, val in zip(other_vars, combo):
                prob *= cond_probs[key][v][val]
            new_rows.append([prob, *combo, *key])
    # 4) Output: column names without primes, ordered as (other_vars, target_vars).
    cols = ["Pr"] + other_vars + target_vars
    new_df = pd.DataFrame(new_rows, columns=cols)
    # Renormalise so that probabilities sum to 1.
    new_df["Pr"] = new_df["Pr"] / new_df["Pr"].sum()
    # print(new_df.head())
    return new_df


# ---------------------------------------------------------------------------
# Information-theoretic primitives: entropy / multivariate mutual information /
# conditional entropy / conditional mutual information (all multivariate).
# ---------------------------------------------------------------------------

def _to_df(data):
    """
    Internal helper: return `data` if it is already a DataFrame, otherwise
    read it as an Excel file.
    """
    return data if isinstance(data, pd.DataFrame) else pd.read_excel(data)


def entropy(data, vars: List[str]) -> float:
    """
    Joint entropy H(vars) of the random-variable set `vars`.

    Parameters
    ----------
    data: DataFrame or path; must include the `vars` columns and a 'Pr' column.
    vars: list of variable names; at least one element.

    Returns
    -------
    H = -sum_v p(v) log2 p(v), where p(v) is the probability that the
    `vars` group takes the joint value v.
    """
    df = _to_df(data)
    probs = df.groupby(vars, sort=False)["Pr"].sum().values
    probs = probs[probs > 0]
    return -(probs * np.log2(probs)).sum()


def mutual_information(data, vars: List[str]) -> float:
    """
    Multivariate "intersection information" I(X1; ... ; Xn), computed via
    the inclusion-exclusion principle:

        I(vars) = sum_{k=1..n} (-1)^(k-1) * sum_{U subset of vars, |U|=k} H(U)

    Parameters
    ----------
    data: DataFrame or path containing the `vars` columns and a 'Pr' column.
    vars: list of variable names participating in the calculation; len >= 1.

    Returns
    -------
    Interaction information (float).
    """
    if not isinstance(vars, list) or len(vars) < 1:
        raise ValueError("`vars` must be a list with at least one element")
    result = 0.0
    n = len(vars)
    # Inclusion-exclusion principle.
    for k in range(1, n + 1):
        for subset in itertools.combinations(vars, k):
            H_sub = entropy(data, list(subset))
            result += ((-1) ** (k - 1)) * H_sub
    return result


def conditional_entropy(data, target: str, given: List[str]) -> float:
    """
    Conditional entropy H(target | given).

    Parameters
    ----------
    data  : DataFrame or path; must contain the target and given columns,
            plus a 'Pr' column.
    target: target variable name.
    given : list of conditioning variables.

    Returns
    -------
    H(target | given) = H(target ∪ given) - H(given)
    """
    return entropy(data, [target] + given) - entropy(data, given)


def conditional_mutual_information(data, target: str, xs: List[str], given: List[str]) -> float:
    """
    Conditional mutual information I(xs; target | given).

    Parameters
    ----------
    data  : DataFrame or path.
    target: target variable name (or list).
    xs    : list of variables whose conditional MI with target is measured.
    given : list of conditioning variables.

    Returns
    -------
    I(xs; target | given) = H(target | given) - H(target | given ∪ xs)
    """
    return conditional_entropy(data, target, given) - conditional_entropy(data, target, given + xs)

# ---------------------------------------------------------------------------
# Two-source PID: takes a 2-element src list and a target, returns a pd.Series.
# ---------------------------------------------------------------------------
def two_source_pid(df: DataFrame, src: List[str], tgt: str) -> pd.Series:
    if len(src) != 2:
        raise ValueError("`src` must be of length 2")
    src1, src2 = src
    order = len(src) - 1
    ndf = df_to_new_df(df, src, tgt, order)
    # Local edit retained for compatibility with the Ising-model script.
    q_df = build_conditionally_independent_df(ndf, [tgt])
    col1 = f"({src1})"
    col2 = f"({src2})"
    redundancy = mutual_information(q_df, [col1, col2])
    h_q = conditional_entropy(q_df, tgt, [col1, col2])
    h_p = conditional_entropy(df, tgt, [src1, src2])
    synergy = h_q - h_p
    unique_src1 = conditional_mutual_information(q_df, tgt, [col1], [col2])
    unique_src2 = conditional_mutual_information(q_df, tgt, [col2], [col1])
    result = pd.Series({
        'redundancy':   redundancy,
        'synergy':      synergy,
        'unique_src1':  unique_src1,
        'unique_src2':  unique_src2,
    })
    result.name = (src1, src2, tgt)
    return result
# ---------------------------------------------------------------------------
# Total synergistic effect (standalone).
# ---------------------------------------------------------------------------
def total_syn_effect(df: DataFrame, src: List[str], tgt: str) -> float:
    """
    Total synergistic gain on the order=1 distribution:
      total_syn_effect = H_Q1(tgt | all combination columns) - H_Q0(tgt | all src)
    where:
      Q1 = build_conditionally_independent_df(df_to_new_df(df, src, tgt, order=1), [tgt])
      Q0 = build_conditionally_independent_df(df, [tgt])

    Returns: float.
    """
    # print(df_to_new_df(df, src, tgt, order=1))
    q_df_1 = build_conditionally_independent_df(df_to_new_df(df, src, tgt, order=1), [tgt])
    H_new = conditional_entropy(q_df_1, tgt, [col for col in q_df_1.columns if col not in ['Pr', tgt]])
    H_old = conditional_entropy(df, tgt, src)
    # print(H_new)
    return H_new - H_old

# ---------------------------------------------------------------------------
# Multi-source synergistic effect and per-order synergies.
# ---------------------------------------------------------------------------
def multi_source_syn(df: DataFrame, src: List[str], tgt: str) -> pd.Series:
    """
    Multi-source synergistic information, returned as a pd.Series with the
    fields n, total_syn_effect, and the per-order synergistic effects:
      - 'n'                : number of source variables
      - 'total_syn_effect' : H_Q - H_P based on the order=1 expansion
      - 'order_k_syn'      : per-order synergistic effect, k = 1 .. n

    For each k:
      1) df_prev_raw = df_to_new_df(df, src, tgt, order=k-1) or the original df
      2) df_prev = build_conditionally_independent_df(df_prev_raw, [tgt])
      3) df_curr_raw = df_to_new_df(df, src, tgt, order=k) or the original df
      4) df_curr = build_conditionally_independent_df(df_curr_raw, [tgt])
      5) H1 = H(tgt | all combination columns) on df_prev
      6) H2 = H(tgt | all combination columns) on df_curr
      order_k_syn = H1 - H2

    total_syn_effect computation:
      Q_df_1 = build_conditionally_independent_df(df_to_new_df(df, src, tgt, order=1), [tgt])
      H_new = H(tgt | all combination columns) on Q_df_1
      Q_df_0 = build_conditionally_independent_df(df, [tgt])
      H_old = H(tgt | src) on Q_df_0
      total_syn_effect = H_new - H_old

    Returns
    -------
    pd.Series with index ['n', 'total_syn_effect', 'order_1_syn', ...,
                          'order_n_syn'] and `name = (tuple(src), tgt)`.
    """
    n = len(src)
    results: dict = {'n': n}
    # Per-order synergistic effects.
    for k in range(n, n + 1):
        # H1 distribution.
        if k > 1:
            df_prev_raw = df_to_new_df(df, src, tgt, order=k - 1)
            df_prev = build_conditionally_independent_df(df_prev_raw, [tgt])
        else:
            df_prev = build_conditionally_independent_df(df, [tgt])
        H1_vars = [col for col in df_prev.columns if col not in ['Pr', tgt]]
        # H2 distribution.
        if k < n:
            df_curr_raw = df_to_new_df(df, src, tgt, order=k)
            df_curr = build_conditionally_independent_df(df_curr_raw, [tgt])
        else:
            df_curr = df
        H2_vars = [col for col in df_curr.columns if col not in ['Pr', tgt]]
        # Conditional-entropy difference.
        H1 = conditional_entropy(df_prev, tgt, H1_vars)
        H2 = conditional_entropy(df_curr, tgt, H2_vars)
        results[f'order_{k}_syn'] = H1 - H2
        print(results)
    series = pd.Series(results)
    series.name = (tuple(src), tgt)
    return series


def multi_source_un(df: pd.DataFrame, src: List[str], tgt: str) -> pd.Series:
    """
    Per-source unique information about the target.

    Parameters
    ----------
    df : pd.DataFrame
        Original joint-probability DataFrame containing the columns
        src... , tgt and 'Pr'.
    src : List[str]
        List of source variable names, e.g. ['S1','S2','S3'].
    tgt : str
        Target variable name, e.g. 'T'.

    Returns
    -------
    pd.Series
        Indexed by "Un(source -> target | other-sources combination)",
        with the corresponding unique-information values.
        Series.name = (tuple(src), tgt)
    """
    results: dict[str, float] = {}

    # 1) Cast every variable's value to an integer-string for clean joining.
    df_str = df.copy()
    for v in src + [tgt]:
        df_str[v] = df_str[v].astype(int).astype(str)

    # 2) For each source variable, compute its unique information.
    for s in src:
        # 2.1) Build the list of "other" source variables.
        others = [v for v in src if v != s]
        # 2.2) Synthesise a column name for the other-source group, e.g. '(S2,S3,S4)'.
        oth_col = f"({','.join(others)})"

        # 2.3) Generate the merged intermediate table on df_str.
        df_m = df_str.copy()
        # Concatenate the other-source values into the new column.
        df_m[oth_col] = df_m.apply(
            lambda row: ",".join(row[v] for v in others),
            axis=1
        )
        # Aggregate into a three-variable joint distribution: s, oth_col, tgt.
        df_m2 = (
            df_m
            .groupby([s, oth_col, tgt], sort=False)["Pr"]
            .sum()
            .reset_index()
        )

        # 2.4) Force conditional independence: given tgt, make s and oth_col independent.
        # print(df_m2.head())
        q_df = build_conditionally_independent_df(df_m2, [tgt])

        # 2.5) Compute the conditional MI I(tgt; s | oth_col)
        #      = H(tgt | oth_col) - H(tgt | [s, oth_col]).
        unq_val = conditional_mutual_information(
            q_df,
            tgt,
            [s],
            [oth_col]
        )

        # 2.6) Store the result; the index name encodes the specific
        #      "other sources" combination.
        results[f"Un({s}->{tgt}|{oth_col})"] = unq_val

    # 3) Return the named Series so multiple calls can be merged into a
    #    MultiIndex DataFrame.
    ser = pd.Series(results)
    ser.name = (tuple(src), tgt)
    return ser

# ---------------------------------------------------------------------------
# Multi-source redundancy (n >= 2).
# ---------------------------------------------------------------------------
def multi_source_red(df: pd.DataFrame, src: List[str], tgt: str) -> float:
    """
    Multi-source redundancy.

    Steps (same shape as total_syn_effect):
      1) Apply the order=1 transform to the original distribution df,
         producing a new table with one column per single source '(Si)':
           df1_raw = df_to_new_df(df, src, tgt, order=1)
      2) Force these '(Si)' columns to be conditionally independent given tgt:
           q_df_1 = build_conditionally_independent_df(df1_raw, [tgt])
      3) Take the multivariate interaction information across all '(Si)':
           R = I( (S1); (S2); ... ; (Sn) )

    Returns
    -------
    Redundancy (float).
    """
    # 1) order = 1 -> build the per-source columns '(Si)'.
    df1_raw = df_to_new_df(df, src, tgt, order=1)

    # 2) Force conditional independence given tgt.
    q_df_1 = build_conditionally_independent_df(df1_raw, [tgt])

    # 3) Take all '(Si)' columns and compute the multivariate interaction information.
    combo_cols = [c for c in q_df_1.columns if c.startswith('(') and c.endswith(')')]
    redundancy = mutual_information(q_df_1, combo_cols)

    return redundancy



if __name__ == "__main__":

    var_names = ["S1", "S2", "S3", "S4", "T"]
    src_vars = ["S1", "S2", "S3", "S4"]
    tgt_var = "T"

    neg_cases: List[pd.DataFrame] = []  # collect distributions exhibiting negative redundancy
    for i in range(10000):
        df_rand = generate_random_df(var_names)
        R = multi_source_red(df_rand, src_vars, tgt_var)

        # Print and archive every distribution whose redundancy turns negative.
        if R < -0.0001:
            print(f"\n### Case {i:04d}  -- redundancy = {R:.6f}")
            print(df_rand.to_string(index=False))
            neg_cases.append(df_rand)

    print(f"\nGenerated 10000 random distributions; "
          f"{len(neg_cases)} of them have negative redundancy.")
