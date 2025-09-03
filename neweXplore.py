# General imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.neighbors import NearestNeighbors
import difflib
import math
from collections import OrderedDict

# --- rdkit & py3dmol functions ---
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski, Crippen
import py3Dmol

# -------------------- DATA LOADING --------------------

@st.cache_data(show_spinner=False)
def load_data(features_file, library_file):
    try:
        features = pd.read_excel(features_file, sheet_name='polyFeature', index_col=0)
        properties = pd.read_excel(features_file, sheet_name='polyIndex', index_col=0)
        library = pd.read_excel(library_file, sheet_name='reference', index_col=0)
        return features, properties, library
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

@st.cache_data(show_spinner=False)
def load_smiles(smiles_file):
    try:
        df = pd.read_excel(smiles_file, sheet_name='SMILES')
        if 'Polymer' not in df.columns:
            raise KeyError("'Polymer' column not found in SMILES sheet")
        df.set_index('Polymer', inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading SMILES data: {e}")
        return pd.DataFrame()

# -------------------- Load datasets ----------------------

features, properties, library = load_data(
    "polyeXplore model feature & index.xlsx", 
    "polyeXplore polymer library data.xlsx"
)

smiles_df = load_smiles("polyeXplore_smiles.xlsx")


if features is None or properties is None or library is None or smiles_df.empty:
    st.stop()  # Stop execution if data missing


# --------------------- PREPARE INDICES  ----------------------------

ppi = pd.DataFrame(features.values.dot(properties.values),
                   index=features.index,
                   columns=properties.columns)
ppi_mean, ppi_std = ppi.mean(), ppi.std(ddof=0)
ppi_z = (ppi - ppi_mean) / ppi_std


# ================================ 1 HEADER & INTRODUCTION =================================

st.title("polyeXplore : a polymer exploration model")
st.write("'It is the lone worker who makes the first advance in a subject, the details may be worked out by a team' - Alexander Fleming - 1881–1955")
st.write("Welcome to Polymer eXploration Model!.")
# st.markdown("© polyeXplore — Sibabrata De", unsafe_allow_html=True)

# 👉 Section 1 expander 
with st.expander("1. Introduction", expanded=True):   # expanded=True keeps it open initially
    st.header("Polymer structural features & polymer properties")

    st.markdown("### Introduction")

    st.markdown(
        """
        <div style='text-align: justify;'>
        <p>
        <i>'polyeXplore'</i>, a polymer exploration model, has been developed to visualize 
        key structural features of polymers based on their repeat units and understand how these features influence 
        various polymer properties. Unlike classical approaches that decompose a polymer into fragment contributions, 
        this model looks at the repeat unit holistically—capturing structural features in the way a user intuitively perceives them. 
        </p>
        <p>
        The influence of these structural features on polymer properties is then displayed 
        through a simplified macro-level model. This complements the well-established 
        <i>'Group Contribution Addition'</i>, originally proposed by Herman F. Mark, 
        systematized by D.W. van Krevelen, and later extended by Jozef Bicerano into a comprehensive predictive framework. 
        While group contribution calculates polymer properties by summing fragmental values, 
        <i>polyeXplore</i> quantifies user-defined features of the entire repeat unit—offering a new, intuitive route 
        to exploring structure–property relationships. 
        </p>
        <p>
        In practice, the user provides feature scores for a repeat unit, and the model visualizes their weighted impact 
        on polymer properties. This enables construction of correlation matrices, ranking of polymers within property hierarchies, 
        and identification of nearest equivalents—deepening understanding of how architecture governs performance. 
        </p>
        </div>

        """,
        unsafe_allow_html=True
    )


# --------------------------- DATA OVERVIEW -----------------------------
# 👉 Section 2 expander 

with st.expander("2. Polymer Input + 3D Visualization"):
    

    st.header("Visualization of polymer structural features and its influence on property")

    st.markdown(
    """
    <div style='text-align: justify;'>
    Enter a polymer name to visualize 3D structure of its repeat unit. A list of key structural features
    of the polymer repeat unit is listed here along with key properties which are infuenced by it. It also compares key molecular 
    descriptors of the polymer using RDkit & py3Dmol with polyethylene as reference which helps to understand
    how the polymer repeat unit structure influences its properties.
    </div>
    """,
    unsafe_allow_html=True
)

    st.markdown("### 📊 Key structural features (SF) of polymer repeat Unit (RU)")
    st.markdown(", ".join(features.columns))

    st.markdown("### 📊 Polymer properties influenced by SF's of polymer RU")
    st.markdown(", ".join(properties.columns))

# ---------------------------------- USER INPUT -------------------------------------

    st.markdown("### 📊 Polymer - user input for 3D visualization and dispaly of molecular descriptors")

    # st.markdown("Enter the polymer name and quantify the structural features of its repeat unit below.")

    # Initialize session state variables
    if "polymer" not in st.session_state:
        st.session_state.polymer = ""
    if "user_vals" not in st.session_state:
        st.session_state.user_vals = {c: 0.0 for c in features.columns}
    if "input_saved" not in st.session_state:
        st.session_state.input_saved = False

# ============================= 2. Polymer Input + RDKIT 3D + DESCRIPTOR =====

with st.form("polymer_input_form"):
    polymer_name_input = st.text_input(
        "Enter the polymer name (CAPITAL letters & ##, e.g., POM, PA66):", 
        value=st.session_state.polymer
    ).strip()

    submitted_polymer = st.form_submit_button("Submit")

if submitted_polymer:
    if polymer_name_input:
        if polymer_name_input in smiles_df.index:
            smile_str = smiles_df.loc[polymer_name_input, "SMILE_repeating_unit"]
            mol = Chem.MolFromSmiles(smile_str)

            if mol:
                mol_3d = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())
                AllChem.UFFOptimizeMolecule(mol_3d)

                mol_block = Chem.MolToMolBlock(mol_3d)
                view = py3Dmol.view(width=600, height=400)
                view.addModel(mol_block, 'mol')
                view.setStyle({'stick': {}})
                view.zoomTo()

                # Save viewer HTML + caption in session state
                st.session_state.mol_view_html = view._make_html()
                st.session_state.polymer_caption = f"3D Structure of {polymer_name_input} Repeat Unit"

                st.markdown(
                    f"<p style='text-align: center; font-weight: 500;'>3D Structure of {polymer_name_input} Repeat Unit</p>",
                    unsafe_allow_html=True
                )
                bordered_html = f'''
                <div style="border: 2px solid #888; border-radius: 10px; 
                            padding: 8px; width: 620px; margin: auto;">
                {view._make_html()}
                </div>
                '''
                st.components.v1.html(bordered_html, height=450)

                if "Description" in smiles_df.columns:
                    desc = smiles_df.loc[polymer_name_input, "Description"]
                    if pd.notna(desc):
                        st.markdown(f"**Polymer Description:** {desc}")
            else:
                st.error("Invalid SMILES string for this polymer.")
        else:
            st.warning("Polymer name not found in SMILES database.")

# -------------- RDKIT DESCRIPTOR ANALYSIS PE vs SUBMITTED POLYMER -------------- 

        def calculate_properties(smiles):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return [None]*8  # Always 8 fields!
                molwt = Descriptors.MolWt(mol)
                tpsa = rdMolDescriptors.CalcTPSA(mol)
                logp = Crippen.MolLogP(mol)
                rot_bonds = Lipinski.NumRotatableBonds(mol)
                hetero_atoms = Descriptors.NumHeteroatoms(mol)
                hba = Lipinski.NumHAcceptors(mol)
                hbd = Lipinski.NumHDonors(mol)
                num_rings = rdMolDescriptors.CalcNumRings(mol)
                return [molwt, tpsa, logp, rot_bonds, hetero_atoms, hba, hbd, num_rings]
            except Exception:
                return [None]*8
        
        prop_names = [
            "Molecular Weight (g/mol)", "Topological polar surface area(TPSA) (Å²)", "LogP", 
            "Rotatable Bonds", "Heteroatoms", "H-Bond Acceptors", 
            "H-Bond Donors", "Number of Rings"
        ]

        ref_polymer = "PE"
        ref_smiles = smiles_df.loc[ref_polymer, "SMILE_repeating_unit"]
        ref_props = calculate_properties(ref_smiles)

        sel_polymer = polymer_name_input
        sel_smiles = smiles_df.loc[sel_polymer, "SMILE_repeating_unit"]
        sel_props = calculate_properties(sel_smiles)

        comp_df = pd.DataFrame(
            {ref_polymer: ref_props, sel_polymer: sel_props}, 
            index=prop_names
        )

        st.markdown("### Molecular Descriptor Comparison (Repeat Unit)")
        st.markdown(
            """
            <div style='text-align: justify;'>
            Molecular descriptors based on RDKit which presents fundamental molecular properties based on 
            classical group contribution theories and set the stage for interactive exploration of feature vs property.
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:16px; font-weight:500; color:black;'>Table - Molecular Descriptor using RDKit</p>", 
            unsafe_allow_html=True
        )
        st.markdown(comp_df.to_markdown())
        st.markdown("Note: Descriptors are calculated for the repeat unit only, "
                    "not the full polymer chain. A comparison with PE oligomer is provided as a reference.")
    else:
        st.error("Invalid SMILES string for this polymer.")
else:
    st.warning("Polymer name not found in SMILES database. Please check spelling.")



# ======================== 3. USER FEATURE INPUT & COMPARISON ========================
# 👉 Section 3 exapnder (feature input form, comparison table, bar chart )

with st.expander("3. User Feature Input & Comparison"):
    st.markdown("### 📊 Structural features of repeat units of polymers - exploration of influence vector")
    st.markdown(
        """
        <div style='text-align: justify;'>
        Here user submits scores for the structural features of a polymer. A score for each feature in the 
        range of -5 to +5 is adopted for assessing the weightage on property, except chain flexibility 
        (can be +10 for highly flexible backbones like PE). The weightage reflects the influence of each feature
        on the polymer's properties and generates property hierarchy similar to polymer pyramid.
        Input values are also compared against dataset values, enabling visualization 
        of how it aligns with indicative reference data.
        </div>
        """,
        unsafe_allow_html=True
    )

    valid_polymers = [str(p).upper() for p in features.index]
    dataset_cols = list(features.columns)

    def show_comparison():
        polymer = st.session_state.polymer
        user_vals = st.session_state.user_vals
        user_series = pd.Series(user_vals, name=f"{polymer}-USER")
        orig_series = features.loc[polymer, dataset_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        orig_series.name = f"{polymer}-DATA"
        compare_df = pd.concat([orig_series, user_series], axis=1)

        st.subheader("📊 Feature scores — user input vs dataset")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:16px; font-weight:500; color:black;'>Table - Feature Scores</p>", 
            unsafe_allow_html=True
        )
        st.dataframe(compare_df)

        y = np.arange(len(compare_df))
        h = 0.35
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(y - h/2, compare_df.iloc[:, 0], h, label=compare_df.columns[0])
        ax.barh(y + h/2, compare_df.iloc[:, 1], h, label=compare_df.columns[1])
        ax.set_xlabel("Feature Value", fontsize=8)
        ax.set_title(f"Structural Features: {polymer} — user vs dataset", fontsize=8, loc="left")
        ax.set_yticks(y)
        ax.set_yticklabels(compare_df.index, fontsize=8)
        ax.invert_yaxis()
        ax.tick_params(axis="x", labelsize=8)
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)

    # --- Feature input form ---
    if st.session_state.input_saved:
        st.success(f"✅ Inputs saved for **{st.session_state.polymer}**.")
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Edit inputs"):
                st.session_state.input_saved = False
        with c2:
            if st.button("Reset inputs"):
                st.session_state.polymer = ""
                st.session_state.user_vals = {c: 0.0 for c in features.columns}
                st.session_state.input_saved = False

        if st.session_state.input_saved and st.session_state.polymer:
            show_comparison()

    else:
        with st.form("user_entry_form", clear_on_submit=False):
            st.markdown("**Enter structural feature values for the polymer:**")
            cols_per_row = 4
            for i, fname in enumerate(dataset_cols):
                if i % cols_per_row == 0:
                    row = st.columns(cols_per_row)
                col = row[i % cols_per_row]
                st.session_state.user_vals[fname] = col.number_input(
                    label=fname,
                    value=float(st.session_state.user_vals.get(fname, 0.0)),
                    step=0.1,
                    format="%.3f",
                    key=f"feat_{fname}"
                )
            submitted_features = st.form_submit_button("Save")
            if submitted_features:
                polymer = polymer_name_input.upper().strip()
                if not polymer:
                    st.error("Please enter a polymer name above.")
                elif polymer not in valid_polymers:
                    st.error("❌ Polymer not found in dataset index.")
                    sugg = difflib.get_close_matches(polymer, valid_polymers, n=5, cutoff=0.6)
                    if sugg:
                        st.caption("Did you mean: " + ", ".join(sugg))
                else:
                    st.session_state.polymer = polymer
                    st.session_state.input_saved = True
                    st.success(f"✅ Saved inputs for **{polymer}**.")
                    show_comparison()


# ============================== 4. NEAREST EQUIVALENT POLYMERS ================================
# ============================== 4. NEAREST EQUIVALENT POLYMERS ================================
# 👉 Section 4 expander (NN search, plots, features table, properties table)

with st.expander("4. Nearest Equivalent Polymers"):
        
    st.markdown("### 📊 Nearest equivalent polymers based on structural features input")

    st.markdown(
        """
        <div style='text-align: justify;'>
        This section identifies polymers that are most similar to the selected polymer 
        based on a simple Polymer Property Index (PPI) values deduced from features and its influence on property.  
        The nearest neighbors are found using a similarity search in feature space, 
        and results are shown.
        </div>
        """,
        unsafe_allow_html=True
    )

    if not st.session_state.get("input_saved"):
        st.info("Please complete and save the user entry above to compute nearest polymers.")
    else:
        polymer = st.session_state.polymer

        # Build user feature vector aligned to dataset order
        feature_order = list(features.columns)
        user_feat = pd.Series(st.session_state.user_vals, dtype=float).reindex(feature_order)

        # Compute user property index (PPI) and Z-score normalized PPI
        user_ppi = user_feat.dot(properties)
        user_ppi.name = f"{polymer}-user"
        user_ppi_z = (user_ppi - ppi_mean) / ppi_std

        # Prepare data for nearest neighbors search in PPI z-score space
        props_for_nn = list(ppi_z.columns)
        X = ppi_z[props_for_nn].astype(float).values
        y = user_ppi_z[props_for_nn].astype(float).values.reshape(1, -1)

        # NearestNeighbors model to find 3 closest polymers
        nn = NearestNeighbors(n_neighbors=3, metric="euclidean")
        nn.fit(X)
        distances, indices = nn.kneighbors(y)  # distances and indices have shape (1, n_neighbors)

        # Flatten arrays
        distances = distances.flatten()
        indices = indices.flatten()

        # Extract neighbor names and distances
        neighbor_names = np.array(ppi_z.index)[indices]
        neighbor_dist = distances

        # Sort neighbors by ascending distance
        order = np.argsort(neighbor_dist)
        names_ordered = neighbor_names[order]
        dists_ordered = neighbor_dist[order]

        # Distance of USER polymer vs dataset baseline
        from sklearn.metrics import euclidean_distances
        user_vs_data_dist = euclidean_distances(
            user_ppi_z.values.reshape(1, -1),
            ppi_z.loc[[polymer]].values
        )[0][0]

        # ---------------- Nearest Polymers Chart ----------------
        st.subheader("1. Nearest equivalent polymers based on input features")

        chart_names = [f"{polymer}-DATA", f"{polymer}-USER"] + list(names_ordered)
        chart_dists = [0.00, round(user_vs_data_dist, 3)] + list(dists_ordered)
        colors = ["darkgreen", "orange"] + ["steelblue"] * len(names_ordered)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(chart_names[::-1], chart_dists[::-1], color=colors[::-1])
        ax.set_xlabel("Relative distance in PPI space (min = 0)")
        ax.set_ylabel("Polymer")
        ax.set_title("Nearest polymers (User vs Reference)")
        for y_i, v in enumerate(chart_dists[::-1]):
            ax.text(v, y_i, f" {v:.2f}", va="center")
        fig.tight_layout()
        st.pyplot(fig)

        st.caption("Note: Distances are relative to the dataset polymer’s PPI (baseline = 0.00).")

        # --- Structural Features of Nearest Polymers ---
        st.subheader("2. Structural features of nearest equivalent polymers in feature space")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:16px; font-weight:500; color:black;'>Table - Structural Features of nearest polymers</p>", 
            unsafe_allow_html=True
        )

        # Avoid duplicating user's polymer if present in nearest neighbor list
        nn_list = [p for p in names_ordered if p != polymer]

        # Extract feature data for nearest neighbors and user input
        nearest_poly_features = features.loc[nn_list].T  # Features as rows, polymers as columns
        user_feat.name = f"{polymer}-USER"

        try:
            data_col = features.loc[polymer].rename(f"{polymer}-DATA")
        except KeyError:
            st.error(f"Polymer '{polymer}' not found in dataset index.")
            data_col = pd.Series(index=features.columns, dtype=float, name=f"{polymer}-DATA")

        # Combine user input, dataset data, and nearest neighbors for comparison
        compare_T = pd.concat(
            [user_feat.to_frame(), data_col.to_frame(), nearest_poly_features],
            axis=1
        )

        st.dataframe(compare_T.style.format("{:.2f}"))

        # --- Properties of Nearest Polymers from Reference Library ---
        st.subheader("3. Properties of nearest polymers from reference library")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:16px; font-weight:500; color:black;'>Table - Polymer property from reference</p>", 
            unsafe_allow_html=True
        )

        # Ensure no duplicates: combine selected polymer + neighbors, but drop duplicates
        unique_polymers = [polymer] + [p for p in names_ordered if p != polymer]

        neighbor_library = library.loc[unique_polymers].copy()
        neighbor_library_T = neighbor_library.T
        st.dataframe(neighbor_library_T.style.format("{:.2f}"))

        st.markdown("Note: The nearest polymers are determined based on Euclidean distance "
                    "in the standardized property index space, with the selected polymer’s "
                    "library data shown for reference.")


# =========================== 5. POLYMER PROPERTY HIERARCHY ============================
# 👉 Section 5 expander (category multiselect, property heatmap)
#section 5 expander (category multiselect, property heatmap)
with st.expander("5. Polymer Property Hierarchy Heatmap"):
    
    st.markdown("### Polymer property hierarchy by category")
    st.markdown(
    """
    <div style='text-align: justify;'>
    This section explores how polymers belonging to commodity, engineering, and high-performance 
    categories differs in hierarchy of properties. A heatmap visualizes the properties, enabling compare groups 
    side by side. User can select polymers from each category and pick properties to display.
    </div>
    """,
    unsafe_allow_html=True
)

    # Define categories and their polymers (in desired order)
    categories_ordered = [
        ("Commodity", ["PE", "PP", "PS", "PVC", "ABS", "PMMA", "ASA"]),
        ("Engineering", ["PA12", "PA6", "PA66", "PBT", "PET", "POM-H", "POM-C",
                        "POK", "COC", "PC", "PAR", "PPE", "PA46", "PPS", "PARA"]),
        ("High-performance", ["PPA", "PEI", "PES", "PSU", "PAI", "PI", "LCP", "PEEK", "PEK", "PBI"]),
    ]

    # Filter only polymers available in the dataset
    available = set(ppi.index)
    categories = [(cat, [p for p in plist if p in available]) for cat, plist in categories_ordered]

    # Initialize session state for selection controls
    for cat_name, _ in categories:
        key = f"sel_{cat_name.lower().replace(' ', '_').replace('-', '_')}"
        if key not in st.session_state:
            st.session_state[key] = []

    # Top-level control buttons for quick select/clear all
    c1, c2 = st.columns(2)
    if c1.button("Select ALL"):
        for cat_name, plist in categories:
            st.session_state[f"sel_{cat_name.lower().replace(' ', '_').replace('-', '_')}"] = plist[:]
    if c2.button("Clear ALL"):
        for cat_name, _ in categories:
            st.session_state[f"sel_{cat_name.lower().replace(' ', '_').replace('-', '_')}"] = []

    # Display categories in three columns with selection controls
    cols = st.columns(3)
    selected_polymers_all = []
    for (cat_name, plist), col in zip(categories, cols):
        key = f"sel_{cat_name.lower().replace(' ', '_').replace('-', '_')}"
        with col:
            st.subheader(cat_name)
            btn_c1, btn_c2 = st.columns(2)
            if btn_c1.button("Select All", key=f"select_all_{key}"):
                st.session_state[key] = plist[:]
            if btn_c2.button("Clear", key=f"clear_{key}"):
                st.session_state[key] = []
            selected = st.multiselect(
                f"Select {cat_name.lower()} polymers",
                options=plist,
                default=st.session_state[key],
                key=key
            )
            selected_polymers_all.append(selected)

    # Combine selected polymers across all categories, preserving order and uniqueness
    from collections import OrderedDict
    sel_polymers = list(OrderedDict.fromkeys(p for group in selected_polymers_all for p in group))

    # Pick property subset (limit to 8 for readability)
    st.markdown("### Pick indicative properties for visualization")
    property_options = list(ppi.columns)
    max_props = 8
    if len(property_options) > max_props:
        property_options = property_options[:max_props]

    if "sel_props" not in st.session_state:
        st.session_state.sel_props = property_options[:]

    col_prop1, col_prop2 = st.columns(2)
    with col_prop1:
        if st.button("Select All Properties"):
            st.session_state.sel_props = property_options[:]
    with col_prop2:
        if st.button("Clear Properties"):
            st.session_state.sel_props = []

    selected_props = st.multiselect(
        "Select properties",
        options=list(ppi.columns),
        default=st.session_state.sel_props,
        key="sel_props"
    )

    # Display Heatmap if selections are valid
    if sel_polymers and selected_props:
        subset_ppi = ppi.loc[sel_polymers, selected_props]
        st.markdown("### Polymer Property Index Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(subset_ppi, annot=True, cmap="RdYlBu", center=0,
                    annot_kws={"size": 8, "color": "grey"}, ax=ax)
        ax.set_xlabel("Property")
        ax.set_ylabel("Polymer")
        ax.set_title("Polymer Property Index — Selected Polymers & Properties")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

    else:
        st.info("Please select at least one polymer and one property to display.")


# ========================== 6. RANDOM SELECTIONS & CORRELATIONS OF POLYMER & PROPERTY =========================
# Random selection of polymers and properties
 # 👉 Section 6 expander (random 5 polymers × 3 props heatmap)
with st.expander("6. Random Polymer Rankings"):
   

    import random
    st.markdown("### Random polymer property rankings")
    st.markdown(
    """
    <div style='text-align: justify;'>
    A random sample of polymers and properties is shown to demonstrate polymer hierarchy based on the model, this also 
    demonstrates accuarcy of the designed model. Refreshing the app generates a new subset for exploratory 
    analysis fitting the polymers as per hierarchy similar to polymer pyramid.
    </div>
    """,
    unsafe_allow_html=True
)
    st.markdown("<br>", unsafe_allow_html=True)
    random_polymers = ppi.sample(n=5)
    random_props = np.random.choice(ppi.columns, size=3, replace=False)
    random_subset = random_polymers.loc[:, random_props]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(random_subset, annot=True, cmap="RdYlBu", center=0,
                annot_kws={"size": 8, "color": "grey"}, ax=ax)
    ax.set_title("Random Selection: Polymer Property Rankings")
    ax.set_xlabel("Property")
    ax.set_ylabel("Polymer")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    # ========================== 7. CORRELATION ANALYSIS OF FEATURES & PROPERTIES =========================

    # Polymer Feature-Property Correlation Matrix
    st.markdown("### Polymer Feature-Property Correlation")
    st.markdown(
    """
    <div style='text-align: justify;'>
    This section analyzes correlations between structural features and properties. 
    A heatmap highlights strong relationships that guide feature–property insights.
    </div>
    """,
    unsafe_allow_html=True
)
    st.markdown("<br>", unsafe_allow_html=True)
    corr_matrix = pd.DataFrame({
        prop: [features[f].corr(ppi[prop]) for f in features.columns]
        for prop in ppi.columns
    }, index=features.columns)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="RdYlBu", center=0,
                annot_kws={"size": 7, "color": "grey"}, ax=ax)
    ax.set_title("Correlation between Structural Features and Polymer Properties")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)


# ========================= 8. POSITIVE NEGATIVE INFLUENCE - FEATURES vs PROPERTY ============================
# 👉 Section 7 expander (random 3×3 subset with styled dataframe)
with st.expander("8. Influence Map"):
    

    # --- Sign map (with random subset) ---
    st.markdown("### Influence Map of structural features vs polymer properties (random selection)")
    st.markdown(
    """
    <div style='text-align: justify;'>
    A simplified map shows whether structural features contribute positively, 
    negatively, or neutrally to selected properties. Random subsets are shown 
    each time to avoid overwhelming detail.
    </div>
    """,
    unsafe_allow_html=True
)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:16px; font-weight:500; color:black;'>Influence Map - graphical</p>", 
        unsafe_allow_html=True
    )

    # Pick random subset of 3 features and 3 properties
    rand_features = np.random.choice(features.columns, size=3, replace=False)
    rand_properties = np.random.choice(ppi.columns, size=3, replace=False)

    corr_subset = corr_matrix.loc[rand_features, rand_properties]

    sign_matrix = corr_subset.applymap(
        lambda x: "Positive" if x > 0 else ("Negative" if x < 0 else "Neutral")
    )

    color_map = {
        "Positive": "background-color: #d73027; color: white;",
        "Negative": "background-color: #4575b4; color: white;",
        "Neutral": "background-color: #ffffbf; color: black;",
    }

    def style_sign_cells(val):
        return color_map.get(val, "")

    styled_sign_matrix = sign_matrix.style.applymap(style_sign_cells)
    st.dataframe(styled_sign_matrix)
    st.caption("Note: Random subset of 3 features × 3 properties is shown. Refresh the app to see another sample.")

       # Pick random 3 properties
    # --- Top 3 structural features influencing property (sample view) ---

    st.markdown("### Top 3 structural features influencing selected properties")
    st.markdown(
    """
    <div style='text-align: justify;'>
    For randomly chosen properties, this section highlights the top 3 structural 
    features with the strongest positive or negative influence. 
    </div>
    """,
    unsafe_allow_html=True
)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:16px; font-weight:500; color:black;'>Influence Map - Features vs Property</p>", 
        unsafe_allow_html=True
    )
    # Pick random 3 properties
    rand_props = np.random.choice(corr_matrix.columns, size=3, replace=False)

    top_features_list = []
    for prop in rand_props:
        series = corr_matrix[prop].dropna()

        # Top 3 by absolute correlation
        top_abs = series.reindex(series.abs().sort_values(ascending=False).index[:3])

        for feat, val in top_abs.items():
            top_features_list.append({
                "Property": prop,
                "Feature": feat,
                "Correlation": round(val, 4),
                "Direction": "Positive" if val > 0 else "Negative" if val < 0 else "Neutral"
            })

    # Convert to DataFrame
    top_features_df = pd.DataFrame(top_features_list)

    # Reset index to drop the 0,1,2... serial numbers
    top_features_df = top_features_df.reset_index(drop=True)

    # Collapse duplicate Property entries for a cleaner grouped look
    df_display = top_features_df.copy()
    df_display["Property"] = df_display["Property"].where(
        df_display["Property"].ne(df_display["Property"].shift()), ""
    )

    # Apply red (positive) / blue (negative) styling
    def color_corr(val):
        if val > 0:
            return "color: red; font-weight:600;"
        elif val < 0:
            return "color: blue; font-weight:600;"
        return ""

    styled_df = df_display.style.applymap(color_corr, subset=["Correlation"])

    st.dataframe(styled_df, use_container_width=True)

    st.caption("Note: Random 3 properties are sampled each time. Refresh app for new selection.")


# ===================== 9. Cumulative Positive/Negative Contribution per Property ========
# 👉 Section 8 expander (overall pos/neg contribution bar chart + detailed subset table)
with st.expander("10. Positive/Negative Contributions"):


    st.markdown("### 📊 Positive / Negative contribution of structural features on property")
    st.markdown(
    """
    <div style='text-align: justify;'>
    A cumulative breakdown of how features contribute across multiple 
    properties. Visualization summarizes positive versus negative influences of features on property.
    </div>
    """,
    unsafe_allow_html=True
)
    st.markdown("<br>", unsafe_allow_html=True)
    # Split correlations into positive and negative contributions per property
    pos_contrib = corr_matrix.clip(lower=0).sum(axis=0)
    neg_contrib = corr_matrix.clip(upper=0).abs().sum(axis=0)

    # Normalize to percentage contributions summing to 100%
    total_contrib = pos_contrib + neg_contrib
    pos_pct = (pos_contrib / total_contrib * 100).fillna(0)
    neg_pct = (neg_contrib / total_contrib * 100).fillna(0)

    # Limit to first 8 properties for clarity
    pos_neg_pct_df = pd.DataFrame({
        "Positive": pos_pct,
        "Negative": neg_pct
    }).loc[pos_pct.index[:8]]

    # Plot stacked horizontal bar chart for overall positive/negative contribution per property
    fig, ax = plt.subplots(figsize=(8, 5))
    pos_neg_pct_df.plot(kind='barh', stacked=True, color=['red', 'yellow'], ax=ax)

    ax.set_xlabel("Contribution (%)")
    ax.set_ylabel("Properties")
    ax.set_title("Relative Positive vs Negative Contribution of structural features to properties")
    ax.legend(loc='lower right')

    fig.tight_layout()
    st.pyplot(fig)

    # --- Detailed contribution for a fixed subset of features and properties ---

    import math

    # Select fixed set of 9 features and 8 properties
    features_subset = features.columns[:9]
    properties_subset = corr_matrix.columns[:8]

    # Subset correlation matrix
    corr_subset = corr_matrix.loc[features_subset, properties_subset]

    # Calculate sum of positive and negative correlations per property
    pos_sum = corr_subset.clip(lower=0).sum(axis=0)
    neg_sum = corr_subset.clip(upper=0).abs().sum(axis=0)

    # Prepare summary dataframe
    summary_df = pd.DataFrame({
        "Positive": pos_sum,
        "Negative": neg_sum
    }, index=properties_subset)

# ============================== 10. CONCLUSION & NEXT STEPS ================================
# 👉 Section 10 expander (conclusion)
with st.expander("11. Conclusion", expanded=True):


    st.markdown("### Conclusion")

    st.markdown(
        """
        <div style='text-align: justify;'>
        <p>
        <i>polyeXplore</i> introduces a fresh perspective to polymer property modeling by treating the repeat unit as a whole 
        rather than reducing it to additive fragments. Where group contribution relies on fragmental summation 
        (e.g., –CH₂– and –CH(Ph)– in polystyrene), <i>polyeXplore</i> enables users to assign feature scores directly 
        to the full repeat unit. This streamlined, feature-centric approach—though simpler—has demonstrated the ability 
        to rank polymers correctly across commodity, engineering, and high-performance classes. 
        </p>
        <p>
        In doing so, the model not only complements traditional methods but also democratizes them, 
        making structure–property reasoning more accessible to polymer enthusiasts and learners. 
        A natural next step may be to strengthen the model with curated experimental data and a broader set of polymer types, 
        enhancing both predictive scope and educational value. 
        </p>
        </div>

        """,
        unsafe_allow_html=True
    )

    st.markdown("<hr style='margin-top: 0; margin-bottom: 4px;'>", unsafe_allow_html=True)
    # st.markdown("<p style='text-align: center; font-size: small; margin-top: 0; margin-bottom: 0;'>***End of Analysis***</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: right; font-size: small;'>PolyeXplore - Polymer Property Visualization © Sibabrata De</p>", unsafe_allow_html=True)

# EOF
