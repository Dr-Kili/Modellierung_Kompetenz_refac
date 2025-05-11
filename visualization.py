"""
Visualisierungsfunktionen für die Kompetenzdynamik
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_nullclines(params, fixed_points, output_path, model_nullclines=None,
                   title=None, K_range=None, S_range=None):

    # Import comp_model only if function isn't provided
    if model_nullclines is None:
        import competence_circuit_analysis as comp_model
        model_nullclines = comp_model.nullclines
        classify_fixed_point = comp_model.classify_fixed_point
    else:
        # If model_nullclines is provided, we need classify_fixed_point too
        from competence_circuit_analysis import classify_fixed_point
    
    # Always use 0.0 as starting point for K-axis
    if K_range is None:
        K_range = [0.0, 1.0]
    
    # Create a fine grid for K
    K_grid = np.linspace(K_range[0], K_range[1], 500)
    
    # Calculate nullclines
    S_from_K, S_from_S = model_nullclines(K_grid, params)
    
    # Remove invalid values (negative or infinite)
    valid_K = np.isfinite(S_from_K) & (S_from_K >= 0)
    valid_S = np.isfinite(S_from_S) & (S_from_S >= 0)
    
    # Determine S-axis range based on nullclines and fixed points
    if S_range is None:
        max_s_nullcline = max(np.max(S_from_K[valid_K]) if np.any(valid_K) else 0,
                           np.max(S_from_S[valid_S]) if np.any(valid_S) else 0)
        max_s_fixpoints = max([fp[1] for fp in fixed_points]) if fixed_points else 0
        max_s = max(max_s_nullcline, max_s_fixpoints) * 1.2
        S_range = [0.0, max_s]
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw nullclines with standard colors
    ax.plot(K_grid[valid_K], S_from_K[valid_K], 'b-', label='ComK Nullcline', linewidth=2)
    ax.plot(K_grid[valid_S], S_from_S[valid_S], 'g-', label='ComS Nullcline', linewidth=2)
    
    # Draw fixed points
    for i, fp in enumerate(fixed_points):
        K, S = fp
        fp_type = classify_fixed_point(K, S, params)
        
        # Define colors and markers for fixed points
        if 'Stabil' in fp_type:
            color = 'g'
            label = f'FP{i+1}: Stable Node ({K:.3f}, {S:.3f})'
        elif 'Sattel' in fp_type:
            color = 'y'
            label = f'FP{i+1}: Saddle Point ({K:.3f}, {S:.3f})'
        elif 'Instabil' in fp_type:
            color = 'r'
            label = f'FP{i+1}: Unstable Node ({K:.3f}, {S:.3f})'
        else:
            color = 'gray'
            label = f'FP{i+1}: {fp_type} ({K:.3f}, {S:.3f})'
        
        ax.plot(K, S, 'o', color=color, markersize=10, label=label)
    
    # Explicitly set axis ranges
    ax.set_xlim(K_range)
    ax.set_ylim(S_range)
    
    # Labels
    ax.set_xlabel('ComK Concentration')
    ax.set_ylabel('ComS Concentration')
    
    # Use standard title with Hill coefficients and parameter values if not provided
    if title is None:
        n = params.get('n', 2)
        p = params.get('p', 5)
        bs = params.get('bs', 0.82)
        bk = params.get('bk', 0.07)
        title = f"Nullclines for n={n}, p={p}, bs={bs:.4f}, bk={bk:.4f}"
    
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Save the diagram
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    
    return fig, ax

def plot_phase_diagram(params, K_range, S_range, title="Phase Diagram", 
                      fixed_points=None, show_vector_field=True, trajectories=None,
                      model_odes=None, model_nullclines=None, classify_fixed_point=None):

    if model_odes is None or model_nullclines is None or classify_fixed_point is None:
        import competence_circuit_analysis as comp_model
        if model_odes is None:
            model_odes = comp_model.model_odes
        if model_nullclines is None:
            model_nullclines = comp_model.nullclines
        if classify_fixed_point is None:
            classify_fixed_point = comp_model.classify_fixed_point
    

    if fixed_points:
        K_max = max([fp[0] for fp in fixed_points]) * 1.5  # 50% buffer
        S_max = max([fp[1] for fp in fixed_points]) * 1.2  # 20% buffer
        
        
        if isinstance(K_range, list) and max(K_range) < K_max:
            K_range = [K_range[0], K_max]
        
        elif isinstance(K_range, np.ndarray) and max(K_range) < K_max:
            K_range = np.linspace(min(K_range), K_max, len(K_range))
            
        
        if isinstance(S_range, list) and max(S_range) < S_max:
            S_range = [S_range[0], S_max]
        elif isinstance(S_range, np.ndarray) and max(S_range) < S_max:
            S_range = np.linspace(min(S_range), S_max, len(S_range))
    
    
    if isinstance(K_range, list):
        K_range = np.linspace(K_range[0], K_range[1], 100)
    if isinstance(S_range, list):
        S_range = np.linspace(S_range[0], S_range[1], 100)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Nullkleinen berechnen
    S_from_K, S_from_S = model_nullclines(K_range, params)
    
    # invalide Werte entfernen
    valid_K = np.isfinite(S_from_K) & (S_from_K >= 0)
    valid_S = np.isfinite(S_from_S) & (S_from_S >= 0)
    
    ax.plot(K_range[valid_K], S_from_K[valid_K], 'b-', label='ComK Nullcline (dK/dt = 0)')
    ax.plot(K_range[valid_S], S_from_S[valid_S], 'g-', label='ComS Nullcline (dS/dt = 0)')
    
    # Vektorfeld
    if show_vector_field:
        K_mesh, S_mesh = np.meshgrid(K_range, S_range)
        dK = np.zeros_like(K_mesh)
        dS = np.zeros_like(S_mesh)
        
        for i in range(len(K_range)):
            for j in range(len(S_range)):
                dK[j, i], dS[j, i] = model_odes(0, [K_mesh[j, i], S_mesh[j, i]], params)
        
        # Normalisierung
        magnitude = np.sqrt(dK**2 + dS**2)
        magnitude = np.where(magnitude == 0, 1e-10, magnitude)  # Prevent division by zero
        dK_norm = dK / magnitude
        dS_norm = dS / magnitude
        
        # Vektorfeld zeichnen
        skip = 5  
        ax.quiver(K_mesh[::skip, ::skip], S_mesh[::skip, ::skip], 
                dK_norm[::skip, ::skip], dS_norm[::skip, ::skip],
                color='lightgray', pivot='mid', scale=30, zorder=0)
    
    # trajektoren zeichnen
    if trajectories is not None:
        for i, traj in enumerate(trajectories):
            K_traj, S_traj = traj
            ax.plot(K_traj, S_traj, '-', linewidth=1, alpha=0.7, 
                   color=plt.cm.rainbow(i/len(trajectories)))  # Different colors
    

    if fixed_points:
       
        colors = {
            "Stabiler Knoten": "go",
            "Stabiler Fokus": "go",
            "Instabiler Knoten": "ro",
            "Instabiler Fokus": "ro",
            "Sattelpunkt": "yo",
            "Zentrum": "co",
            "Nicht-hyperbolisch": "mo",
            "Unklassifiziert": "ko"
        }
        
        for i, fp in enumerate(fixed_points):
            K, S = fp
            if K >= 0 and S >= 0 and K <= max(K_range) and S <= max(S_range):
                fp_type = classify_fixed_point(K, S, params)
                ax.plot(K, S, colors.get(fp_type, "ko"), markersize=10, 
                       label=f'FP{i+1}: {fp_type} ({K:.3f}, {S:.3f})')
    
    ax.set_xlim([min(K_range), max(K_range)])
    ax.set_ylim([min(S_range), max(S_range)])
    ax.set_xlabel('ComK Concentration')
    ax.set_ylabel('ComS Concentration')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True)
    
    return fig, ax

def plot_time_series(t, K, S, threshold=0.5, title="Time Series of Competence Dynamics"):
    """
    Plots time series of ComK and ComS concentrations.
    
    Args:
        t: Time array
        K: ComK concentration array
        S: ComS concentration array
        threshold: Threshold for competence
        title: Title for the plot
        
    Returns:
        tuple: (fig, ax, competence_periods) Figure, axes, and list of competence periods
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(t, K, 'b-', label='ComK')
    ax.plot(t, S, 'g-', label='ComS')

    # Achsen normalisieren
    y_max = max(max(K), max(S)) * 1.1  # 10% buffer
    ax.set_ylim(0, y_max)
    
    # Kompetenz, wenn ComK über Schwellwert
    competence_mask = K > threshold
    competence_periods = []
    
    if competence_mask.any():
        # Find transitions (0->1: start, 1->0: end)
        transitions = np.diff(competence_mask.astype(int))
        start_indices = np.where(transitions == 1)[0]
        end_indices = np.where(transitions == -1)[0]
        

        if competence_mask[0]:
            start_indices = np.insert(start_indices, 0, 0)
        if competence_mask[-1]:
            end_indices = np.append(end_indices, len(competence_mask) - 1)
        
        n = min(len(start_indices), len(end_indices))
        for i in range(n):
            start_t = t[start_indices[i]]
            end_t = t[end_indices[i]]
            competence_periods.append((start_t, end_t))
            ax.axvspan(start_t, end_t, alpha=0.2, color='red')
    

    ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label='Competence Threshold')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Concentration')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True)
    
    if competence_periods:
        competence_durations = [end-start for start, end in competence_periods]
        avg_duration = np.mean(competence_durations)
        text_str = f"Number of competence events: {len(competence_periods)}\nAverage duration (Tc): {avg_duration:.2f}"
        ax.text(0.02, 0.98, text_str, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5),
                verticalalignment='top')
    
    return fig, ax, competence_periods

def plot_stochastic_comparison(stochastic_results, output_dir):
   
    if not stochastic_results:
        print("No stochastic results to visualize")
        return
        
    # Extract data for plotting
    param_ids = list(stochastic_results.keys())
    param_labels = [stochastic_results[pid]['name'] for pid in param_ids]
    median_durations = [stochastic_results[pid]['median_duration'] for pid in param_ids]
    cv_durations = [stochastic_results[pid]['cv_duration'] for pid in param_ids]
    median_rise_times = [stochastic_results[pid]['median_rise_time'] for pid in param_ids]
    cv_rise_times = [stochastic_results[pid]['cv_rise_time'] for pid in param_ids]
    
    # Update Duration and Rise Time comparison (side by side)
    plt.figure(figsize=(14, 6))
    
    # Median Values
    plt.subplot(1, 2, 1)
    bar_width = 0.3
    index = np.arange(len(param_ids))
    
    # Plot median values with stronger colors
    bars1 = plt.bar(index - bar_width/2, median_durations, bar_width, color='blue', 
                  alpha=0.9, label='Median Duration')
    bars2 = plt.bar(index + bar_width/2, median_rise_times, bar_width, color='orange', 
                  alpha=0.9, label='Median Rise Time')
    
    plt.xlabel('Parameter Set')
    plt.ylabel('Time')
    plt.title('Comparison of Median Times')
    plt.xticks(index, param_labels, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Place legend outside the first plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Coefficient of Variation
    plt.subplot(1, 2, 2)
    bars3 = plt.bar(index - bar_width/2, cv_durations, bar_width, color='blue', 
                  alpha=0.7, label='Total Duration')
    bars4 = plt.bar(index + bar_width/2, cv_rise_times, bar_width, color='orange', 
                  alpha=0.7, label='Rise Time')
    plt.xlabel('Parameter Set')
    plt.ylabel('Coefficient of Variation')
    plt.title('Timing Variability')
    plt.xticks(index, param_labels, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Place legend outside the second plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'timing_comparison.png'), bbox_inches='tight')
    plt.close()

def plot_parameter_histograms(excitable_configs, params_base):

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    

    param_names = ['n', 'p', 'ak', 'bk', 'bs', 'k0', 'k1']
    
    standard_params = {
        'n': 2,
        'p': 5,
        'ak': 0.004,
        'bk': 0.07,
        'bs': 0.82,
        'k0': 0.2,
        'k1': 0.222
    }
    

    param_values = {
        name: [cfg[i] for cfg in excitable_configs] 
        for i, name in enumerate(param_names)
    }
    
    
    for i, param in enumerate(param_names):
        if param in param_values and param_values[param]:
            n_bins = min(30, len(param_values[param]) // 5)
            axes[i].hist(param_values[param], bins=n_bins, alpha=0.7)
            axes[i].set_xlabel(param)
            axes[i].set_ylabel('Häufigkeit')
            axes[i].set_title(f'Verteilung von {param}')
            
            std_val = standard_params[param]
            axes[i].axvline(std_val, color='b', linestyle='-', 
                          label=f'Standard: {std_val:.3f}')
            
            if param not in ['n', 'p']:
                mean_val = np.mean(param_values[param])
                median_val = np.median(param_values[param])
                axes[i].axvline(mean_val, color='r', linestyle='--', 
                              label=f'Mittelwert: {mean_val:.3f}')
                axes[i].axvline(median_val, color='g', linestyle=':', 
                              label=f'Median: {median_val:.3f}')
            
            axes[i].legend(fontsize='small')
    

    if excitable_configs:  
        try:
            from competence_circuit_analysis import nullclines
        except ImportError:
            # If unable to import directly, define nullclines function
            def nullclines(K_range, params):
                # Unpack parameters
                ak = params['ak']
                bk = params['bk']
                bs = params['bs']
                k0 = params['k0']
                k1 = params['k1']
                n = params['n']
                p = params['p']
                
                # ComK nullcline: dK/dt = 0 => S as a function of K
                # ak + (bk * K**n) / (k0**n + K**n) - K / (1 + K + S) = 0
                # Solving for S:
                S_from_K = K_range / (ak + (bk * K_range**n) / (k0**n + K_range**n)) - (1 + K_range)
                
                # ComS nullcline: dS/dt = 0 => S as a function of K
                # bs / (1 + (K/k1)**p) - S / (1 + K + S) = 0
                # Using approximation for numerical stability:
                S_from_S = bs * (1 + K_range) / (1 + (K_range/k1)**p - bs)
                
                return S_from_K, S_from_S
        
        # Take a middle parameter set as an example
        sample_config_idx = len(excitable_configs) // 2
        sample_params = params_base.copy()
        
        for i, param_name in enumerate(param_names):
            sample_params[param_name] = excitable_configs[sample_config_idx][i]
        
        K_range = np.linspace(0, 1, 100)
        S_range = np.linspace(0, 10, 100)  # Larger range for ComS
        
        # Redundant, falls Zeit noch anpassen... 
        S_from_K, S_from_S = nullclines(K_range, sample_params)
        
        valid_K = np.isfinite(S_from_K) & (S_from_K >= 0)
        valid_S = np.isfinite(S_from_S) & (S_from_S >= 0)
        
        axes[7].plot(K_range[valid_K], S_from_K[valid_K], 'b-', label='ComK Nullkline')
        axes[7].plot(K_range[valid_S], S_from_S[valid_S], 'g-', label='ComS Nullkline')
        axes[7].set_xlabel('ComK')
        axes[7].set_ylabel('ComS')
        axes[7].set_title('Beispiel Nullklinen')
        axes[7].legend()
    
    plt.tight_layout()
    
    return fig, axes, param_values

def plot_excitable_map(excitable_configs, bs_range, bk_range, output_path, 
                      title="Excitable Configurations"):
    """
    Creates a map of excitable configurations in the bs-bk parameter space.
    
    Args:
        excitable_configs: List of excitable configuration dictionaries
        bs_range: Range of bs values
        bk_range: Range of bk values
        output_path: Path to save the figure
        title: Title for the plot
    """
    plt.figure(figsize=(12, 10))
    
    # Excitable regions in parameter space
    if excitable_configs:
        bs_values = [config['bs'] for config in excitable_configs]
        bk_values = [config['bk'] for config in excitable_configs]
        plt.scatter(bs_values, bk_values, c='red', s=50, alpha=0.7, 
                  label='Excitable Configurations')
    
    # Mark standard values
    import competence_circuit_analysis as comp_model
    std_params = comp_model.default_params()
    plt.scatter([std_params['bs']], [std_params['bk']], c='blue', s=200, marker='*', 
              label='Standard')
    
    plt.xlabel('bs (ComS Expression Rate)')
    plt.ylabel('bk (ComK Feedback Strength)')
    plt.title(title)
    plt.xlim(min(bs_range), max(bs_range))
    plt.ylim(min(bk_range), max(bk_range))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_hill_coefficient_comparison(all_results, output_dir):
  
    n_values = sorted(set([all_results[key]['n'] for key in all_results]))
    p_values = sorted(set([all_results[key]['p'] for key in all_results]))
    
    # Heatmap von erregbaren Konfigurationen
    excitable_counts = np.zeros((len(n_values), len(p_values)))
    
    for i, n in enumerate(n_values):
        for j, p in enumerate(p_values):
            key = f'n{n}_p{p}'
            if key in all_results:
                excitable_counts[i, j] = all_results[key]['excitable_count']
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(excitable_counts, cmap='viridis')

    plt.colorbar(im, label='Number of Excitable Configurations')
    plt.xlabel('p (ComS Repression Hill Coefficient)')
    plt.ylabel('n (ComK Activation Hill Coefficient)')
    plt.title('Comparison of Excitable Region Size')

    plt.xticks(np.arange(len(p_values)), p_values)
    plt.yticks(np.arange(len(n_values)), n_values)
    
    for i in range(len(n_values)):
        for j in range(len(p_values)):
            text = plt.text(j, i, f"{int(excitable_counts[i, j])}", 
                          ha="center", va="center", color="w", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'excitable_count_heatmap.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    
    x_labels = [f'n={n}, p={p}' for n in n_values for p in p_values]
    counts = [all_results[f'n{n}_p{p}']['excitable_count'] 
             for n in n_values for p in p_values]
  
    colors = plt.cm.tab10(np.array([i for i in range(len(n_values)) 
                                 for _ in range(len(p_values))]) % 10)

    bars = plt.bar(np.arange(len(x_labels)), counts, color=colors)

    plt.xlabel('Hill Coefficients (n, p)')
    plt.ylabel('Number of Excitable Configurations')
    plt.title('Size of Excitable Region for Different Hill Coefficients')
    plt.xticks(np.arange(len(x_labels)), x_labels, rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)

    handles = [plt.Rectangle((0,0),1,1, color=plt.cm.tab10(i % 10)) 
              for i in range(len(n_values))]
    labels = [f'n={n}' for n in n_values]
    plt.legend(handles, labels, title='ComK Activation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'excitable_count_barplot.png'), dpi=300)
    plt.close()

def plot_duration_distribution(all_durations, all_rise_times, output_dir):


    plt.figure(figsize=(14, 8))
    bins = np.linspace(0, 50, 25)  
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_durations)))
    
    for i, (param_name, durations) in enumerate(all_durations.items()):
        plt.hist(durations, bins=bins, alpha=0.5, color=colors[i], 
               label=f'{param_name} (n={len(durations)})')
    
    plt.xlabel('Competence Duration')
    plt.ylabel('Frequency')
    plt.title('Distribution of Competence Durations Across Parameter Sets')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'combined_duration_histogram.png'))
    plt.close()
    
    # 2. Combined histogram of rise times
    plt.figure(figsize=(14, 8))
    rise_bins = np.linspace(0, 20, 20)  # Consistent bins for rise times
    
    for i, (param_name, rise_times) in enumerate(all_rise_times.items()):
        plt.hist(rise_times, bins=rise_bins, alpha=0.5, color=colors[i], 
               label=f'{param_name} (n={len(rise_times)})')
    
    plt.xlabel('Rise Time (Time to Maximum ComK)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Rise Times Across Parameter Sets')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'combined_risetime_histogram.png'))
    plt.close()
    
    # 3. Combined scatter plot of rise time vs duration
    plt.figure(figsize=(14, 8))
    
    for i, (param_name, durations) in enumerate(all_durations.items()):
        rise_times = all_rise_times[param_name]
        plt.scatter(rise_times, durations, alpha=0.7, color=colors[i], 
                  label=f'{param_name} (n={len(durations)})')
        
        # Add trendline if we have enough points
        if len(durations) > 3:
            z = np.polyfit(rise_times, durations, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(rise_times), max(rise_times), 100)
            plt.plot(x_range, p(x_range), '--', color=colors[i], alpha=0.7)
    
    plt.xlabel('Rise Time')
    plt.ylabel('Total Duration')
    plt.title('Relationship Between Rise Time and Competence Duration')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'combined_rise_vs_duration.png'))
    plt.close()
    
    plt.figure(figsize=(14, 8))
    
    plt.subplot(1, 2, 1)
    duration_data = [all_durations[param] for param in all_durations.keys()]
    param_labels = list(all_durations.keys())
    plt.boxplot(duration_data, labels=param_labels)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Duration')
    plt.title('Competence Durations by Parameter Set')
    plt.grid(True, alpha=0.3)
    
    # Ansteigszeiten
    plt.subplot(1, 2, 2)
    rise_data = [all_rise_times[param] for param in all_rise_times.keys()]
    plt.boxplot(rise_data, labels=param_labels)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Rise Time')
    plt.title('Competence Rise Times by Parameter Set')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot_comparison.png'))
    plt.close()
    
    # 5. Kernel Density Estimation of Duration Distributions
    plt.figure(figsize=(14, 8))
    
    from scipy.stats import gaussian_kde
    
    x_range = np.linspace(0, 50, 200)
    
    for i, (param_name, durations) in enumerate(all_durations.items()):
        if len(durations) > 5:  # Need enough points for KDE
            # Use Gaussian KDE for smooth density estimation
            kde = gaussian_kde(durations)
            plt.plot(x_range, kde(x_range), '-', linewidth=2, color=colors[i], 
                   label=f'{param_name} (n={len(durations)})')
    
    plt.xlabel('Competence Duration')
    plt.ylabel('Probability Density')
    plt.title('Kernel Density Estimation of Competence Duration Distributions')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'duration_kde.png'))
    plt.close()
    
    # 6. KDE for Rise Times
    plt.figure(figsize=(14, 8))
    
    x_range_rise = np.linspace(0, 20, 200)
    
    for i, (param_name, rise_times) in enumerate(all_rise_times.items()):
        if len(rise_times) > 5:  # Need enough points for KDE
            # Use Gaussian KDE for smooth density estimation
            kde = gaussian_kde(rise_times)
            plt.plot(x_range_rise, kde(x_range_rise), '-', linewidth=2, color=colors[i], 
                   label=f'{param_name} (n={len(rise_times)})')
    
    plt.xlabel('Rise Time')
    plt.ylabel('Probability Density')
    plt.title('Kernel Density Estimation of Rise Time Distributions')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'risetime_kde.png'))
    plt.close()

def plot_parameter_correlations(correlation_df, output_dir):

    plt.figure(figsize=(12, 8))
    
    # Get data from DataFrame
    param_labels = correlation_df['Parameter'].values
    duration_corrs = correlation_df['Correlation with Median Duration'].values
    rise_corrs = correlation_df['Correlation with Median Rise Time'].values
    
    # Create short parameter labels for plotting
    param_labels_short = [label.split(' ')[0] for label in param_labels]
    
    # Duration correlations
    plt.subplot(2, 1, 1)
    bars = plt.bar(param_labels_short, duration_corrs)
    
    # Color bars by sign
    for i, bar in enumerate(bars):
        if duration_corrs[i] > 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.ylabel('Correlation')
    plt.title('Parameter Influence on Median Competence Duration')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Rise time correlations
    plt.subplot(2, 1, 2)
    bars = plt.bar(param_labels_short, rise_corrs)
    
    # Color bars by sign
    for i, bar in enumerate(bars):
        if rise_corrs[i] > 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.ylabel('Correlation')
    plt.title('Parameter Influence on Median Rise Time')
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_correlation_barplot.png'))
    plt.close()

def analyze_parameter_effects(selected_configs, stochastic_results, results_dir):

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Wenn keine Ergebnisse vorhanden sind, abbrechen
    if not stochastic_results:
        print("Keine stochastischen Ergebnisse für die Analyse vorhanden.")
        return
    
    # Unterverzeichnis für die Analyse
    analysis_dir = os.path.join(results_dir, 'parameter_effects')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Daten aus den Ergebnissen extrahieren
    bs_values = []
    bk_values = []
    mean_durations = []
    cv_durations = []
    mean_rise_times = []
    cv_rise_times = []
    init_probabilities = []
    names = []
    
    for i, config in enumerate(selected_configs):
        param_id = f"param_set_{i+1}"
        if param_id in stochastic_results:
            result = stochastic_results[param_id]
            
            bs_values.append(config['bs'])
            bk_values.append(config['bk'])
            names.append(config['name'])
            
            mean_durations.append(result['mean_duration'])
            cv_durations.append(result['cv_duration'])
            mean_rise_times.append(result['mean_rise_time'])
            cv_rise_times.append(result['cv_rise_time'])
            init_probabilities.append(result['init_probability'])
    
    # Parameter-Einflüsse visualisieren
    
    # Durations vs bs und bk
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(bs_values, bk_values, 
               c=mean_durations, s=100, cmap='plasma', alpha=0.8,
               vmin=min(mean_durations) * 0.8 if mean_durations else 0, 
               vmax=max(mean_durations) * 1.2 if mean_durations else 1)
    
    plt.colorbar(scatter, label='Mittlere Kompetenzdauer')
    
    # Labels für jeden Punkt hinzufügen
    for i, name in enumerate(names):
        # Positionen für Labels anpassen, besonders für Eckpunkte
        x, y = bs_values[i], bk_values[i]
        
        # Standard-Offset
        x_offset, y_offset = 5, 5
        ha, va = 'left', 'bottom'
        
        # Speziallabels für besondere Konfigurationen
        special_label = None
        
        # Wenn Name bereits "Starkes ComK Feedback" oder "Niedrige ComS Expression" enthält
        if "Starkes ComK Feedback" in name or "Hohe bk" in name:
            if "Niedrige ComS Expression" in name or "Niedrige bs" in name:
                special_label = "Starkes ComK Feedback\nNiedrige ComS Expression"
                # Für obere linke Ecke
                x_offset, y_offset = 10, -10
                ha, va = 'left', 'top'
        
        elif "Niedrige bk" in name or "Schwaches ComK Feedback" in name:
            if "Hohe bs" in name or "Hohe ComS Expression" in name:
                special_label = "Schwaches ComK Feedback\nHohe ComS Expression"
                # Für untere rechte Ecke
                x_offset, y_offset = -10, 10
                ha, va = 'right', 'bottom'
        
        # Text platzieren mit angepassten Offsets
        if special_label:
            plt.annotate(special_label, (x, y), 
                        xytext=(x_offset, y_offset), textcoords='offset points',
                        ha=ha, va=va, fontweight='bold')
        else:
            plt.annotate(name, (x, y), 
                        xytext=(x_offset, y_offset), textcoords='offset points',
                        ha=ha, va=va)
    
    # Spezielle Punkte markieren als "Ecken"
    if len(bs_values) > 3:
        # Finde extremste Konfigurationen um sie zu beschriften
        max_bk_idx = np.argmax(bk_values)
        min_bk_idx = np.argmin(bk_values)
        max_bs_idx = np.argmax(bs_values)
        min_bs_idx = np.argmin(bs_values)
        
        # Setze Beschriftungen für Extremwerte, falls noch nicht speziell beschriftet
        extremes = [(max_bk_idx, "Hohe\nComK Feedback", (0, -15), 'center', 'top'),
                   (min_bk_idx, "Niedrige\nComK Feedback", (0, 15), 'center', 'bottom'),
                   (max_bs_idx, "Hohe ComS Expression", (15, 0), 'left', 'center'),
                   (min_bs_idx, "Niedrige ComS Expression", (-15, 0), 'right', 'center')]
        
        for idx, label, (x_off, y_off), ha, va in extremes:
            if not any(special in names[idx] for special in ["Starkes", "Schwaches", "Hohe", "Niedrige"]):
                plt.annotate(label, (bs_values[idx], bk_values[idx]), 
                           xytext=(x_off, y_off), textcoords='offset points',
                           ha=ha, va=va, fontsize=9, fontweight='bold')
    
    plt.xlabel('bs (ComS Expressionsrate)')
    plt.ylabel('bk (ComK Feedback-Stärke)')
    plt.title('Einfluss von bs und bk auf die Kompetenzdauer')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(analysis_dir, 'bs_bk_vs_duration.png'), dpi=300)
    plt.close()
    
    # Rise time vs bs und bk
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(bs_values, bk_values, 
               c=mean_rise_times, s=100, cmap='viridis', alpha=0.8,
               vmin=min(mean_rise_times) * 0.8 if mean_rise_times else 0, 
               vmax=max(mean_rise_times) * 1.2 if mean_rise_times else 1)
    
    plt.colorbar(scatter, label='Mittlere Anstiegszeit')
    
    # Labels für jeden Punkt hinzufügen mit verbesserten Positionen
    for i, name in enumerate(names):
        # Positionen für Labels anpassen, besonders für Eckpunkte
        x, y = bs_values[i], bk_values[i]
        
        # Standard-Offset
        x_offset, y_offset = 5, 5
        ha, va = 'left', 'bottom'
        
        # Speziallabels für besondere Konfigurationen
        special_label = None
        
        # Wenn Name bereits "Starkes ComK Feedback" oder "Niedrige ComS Expression" enthält
        if "Starkes ComK Feedback" in name or "Hohe bk" in name:
            if "Niedrige ComS Expression" in name or "Niedrige bs" in name:
                special_label = "Starkes ComK Feedback\nNiedrige ComS Expression"
                # Für obere linke Ecke
                x_offset, y_offset = 10, -10
                ha, va = 'left', 'top'
        
        elif "Niedrige bk" in name or "Schwaches ComK Feedback" in name:
            if "Hohe bs" in name or "Hohe ComS Expression" in name:
                special_label = "Schwaches ComK Feedback\nHohe ComS Expression"
                # Für untere rechte Ecke
                x_offset, y_offset = -10, 10
                ha, va = 'right', 'bottom'
        
        # Text platzieren mit angepassten Offsets
        if special_label:
            plt.annotate(special_label, (x, y), 
                        xytext=(x_offset, y_offset), textcoords='offset points',
                        ha=ha, va=va, fontweight='bold')
        else:
            plt.annotate(name, (x, y), 
                        xytext=(x_offset, y_offset), textcoords='offset points',
                        ha=ha, va=va)
    
    # Spezielle Punkte markieren als "Ecken" - wie bei vorherigem Plot
    if len(bs_values) > 3:
        # Ähnliches Verfahren wie im ersten Plot
        pass
    
    plt.xlabel('bs (ComS Expressionsrate)')
    plt.ylabel('bk (ComK Feedback-Stärke)')
    plt.title('Einfluss von bs und bk auf die Anstiegszeit')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(analysis_dir, 'bs_bk_vs_rise_time.png'), dpi=300)
    plt.close()
    
    # Initiation probability vs bs und bk
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(bs_values, bk_values, 
               c=init_probabilities, s=100, cmap='YlOrRd', alpha=0.8,
               vmin=min(init_probabilities) * 0.8 if init_probabilities else 0, 
               vmax=max(init_probabilities) * 1.2 if init_probabilities else 1)
    
    plt.colorbar(scatter, label='Wahrscheinlichkeit der Initiierung')
    
    # Labels für jeden Punkt hinzufügen - mit ähnlichen Verbesserungen wie oben
    for i, name in enumerate(names):
        x, y = bs_values[i], bk_values[i]
        
        # Standard-Offset
        x_offset, y_offset = 5, 5
        ha, va = 'left', 'bottom'
        
        # Speziallabels für besondere Konfigurationen - wie in den vorherigen Plots
        special_label = None
        
        if "Starkes ComK Feedback" in name or "Hohe bk" in name:
            if "Niedrige ComS Expression" in name or "Niedrige bs" in name:
                special_label = "Starkes ComK Feedback\nNiedrige ComS Expression"
                x_offset, y_offset = 10, -10
                ha, va = 'left', 'top'
        
        elif "Niedrige bk" in name or "Schwaches ComK Feedback" in name:
            if "Hohe bs" in name or "Hohe ComS Expression" in name:
                special_label = "Schwaches ComK Feedback\nHohe ComS Expression"
                x_offset, y_offset = -10, 10
                ha, va = 'right', 'bottom'
        
        # Text platzieren mit angepassten Offsets
        if special_label:
            plt.annotate(special_label, (x, y), 
                        xytext=(x_offset, y_offset), textcoords='offset points',
                        ha=ha, va=va, fontweight='bold')
        else:
            plt.annotate(name, (x, y), 
                        xytext=(x_offset, y_offset), textcoords='offset points',
                        ha=ha, va=va)
    
    plt.xlabel('bs (ComS Expressionsrate)')
    plt.ylabel('bk (ComK Feedback-Stärke)')
    plt.title('Einfluss von bs und bk auf die Initiierungswahrscheinlichkeit')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(analysis_dir, 'bs_bk_vs_initiation.png'), dpi=300)
    plt.close()
    
    # Einzelne Parameter-Einflüsse analysieren
    if len(bs_values) > 1:
        # bs vs Kompetenzdauer
        plt.figure(figsize=(10, 6))
        plt.scatter(bs_values, mean_durations, s=80, c='blue', alpha=0.7)
        
        # Trendlinie hinzufügen
        z = np.polyfit(bs_values, mean_durations, 1)
        p = np.poly1d(z)
        bs_range = np.linspace(min(bs_values), max(bs_values), 100)
        plt.plot(bs_range, p(bs_range), "r--", 
               label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")
        
        # Labels für jeden Punkt hinzufügen
        for i, name in enumerate(names):
            plt.annotate(name, (bs_values[i], mean_durations[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('bs (ComS Expressionsrate)')
        plt.ylabel('Mittlere Kompetenzdauer')
        plt.title('Einfluss der ComS Expressionsrate auf die Kompetenzdauer')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(analysis_dir, 'bs_vs_duration.png'))
        plt.close()
        
        # bk vs Anstiegszeit
        plt.figure(figsize=(10, 6))
        plt.scatter(bk_values, mean_rise_times, s=80, c='green', alpha=0.7)
        
        # Trendlinie hinzufügen
        z = np.polyfit(bk_values, mean_rise_times, 1)
        p = np.poly1d(z)
        bk_range = np.linspace(min(bk_values), max(bk_values), 100)
        plt.plot(bk_range, p(bk_range), "r--", 
               label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")
        
        # Labels für jeden Punkt hinzufügen
        for i, name in enumerate(names):
            plt.annotate(name, (bk_values[i], mean_rise_times[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('bk (ComK Feedback-Stärke)')
        plt.ylabel('Mittlere Anstiegszeit')
        plt.title('Einfluss der ComK Feedback-Stärke auf die Anstiegszeit')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(analysis_dir, 'bk_vs_rise_time.png'))
        plt.close()
        
        # bk vs Initiierung
        plt.figure(figsize=(10, 6))
        plt.scatter(bk_values, init_probabilities, s=80, c='red', alpha=0.7)
        
        # Trendlinie hinzufügen
        z = np.polyfit(bk_values, init_probabilities, 1)
        p = np.poly1d(z)
        bk_range = np.linspace(min(bk_values), max(bk_values), 100)
        plt.plot(bk_range, p(bk_range), "r--", 
               label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")
        
        # Labels für jeden Punkt hinzufügen
        for i, name in enumerate(names):
            plt.annotate(name, (bk_values[i], init_probabilities[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('bk (ComK Feedback-Stärke)')
        plt.ylabel('Initiierungswahrscheinlichkeit')
        plt.title('Einfluss der ComK Feedback-Stärke auf die Initiierungswahrscheinlichkeit')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(analysis_dir, 'bk_vs_initiation.png'))
        plt.close()
    
    # Statistik in Textdatei speichern
    with open(os.path.join(analysis_dir, 'parameter_effects_summary.txt'), 'w') as f:
        f.write("Zusammenfassung der Parametereffekte\n")
        f.write("==================================\n\n")
        
        f.write("Name\tbs\tbk\tKompetenzdauer\tAnstiegszeit\tInitiierung\n")
        for i, name in enumerate(names):
            f.write(f"{name}\t{bs_values[i]:.4f}\t{bk_values[i]:.4f}\t")
            f.write(f"{mean_durations[i]:.2f}\t{mean_rise_times[i]:.2f}\t")
            f.write(f"{init_probabilities[i]:.4f}\n")
        
        # Korrelationskoeffizienten
        if len(bs_values) > 1:
            f.write("\nKorrelationskoeffizienten:\n")
            corr_bs_duration = np.corrcoef(bs_values, mean_durations)[0, 1]
            corr_bs_rise = np.corrcoef(bs_values, mean_rise_times)[0, 1]
            corr_bs_init = np.corrcoef(bs_values, init_probabilities)[0, 1]
            
            corr_bk_duration = np.corrcoef(bk_values, mean_durations)[0, 1]
            corr_bk_rise = np.corrcoef(bk_values, mean_rise_times)[0, 1]
            corr_bk_init = np.corrcoef(bk_values, init_probabilities)[0, 1]
            
            f.write(f"bs vs Kompetenzdauer: {corr_bs_duration:.4f}\n")
            f.write(f"bs vs Anstiegszeit: {corr_bs_rise:.4f}\n")
            f.write(f"bs vs Initiierung: {corr_bs_init:.4f}\n\n")
            
            f.write(f"bk vs Kompetenzdauer: {corr_bk_duration:.4f}\n")
            f.write(f"bk vs Anstiegszeit: {corr_bk_rise:.4f}\n")
            f.write(f"bk vs Initiierung: {corr_bk_init:.4f}\n")