import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pickle

def create_results_directory(prefix="results"):

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_name = f"{prefix}_{timestamp}"

    os.makedirs(dir_name, exist_ok=True)
    
    return dir_name

def print_and_save_parameters(params, results_dir, name="Standard Parameters"):

    print(f"\n{name}:")
    print("="*50)
    
    print("\nCore Parameters:")
    for param_name in ['ak', 'bk', 'bs', 'k0', 'k1', 'n', 'p']:
        if param_name in params:
            print(f"  {param_name}: {params[param_name]}")

    if 'GammaK' in params:
        print("\nExtended Parameters (Suel et al.):")
        for param_name in ['GammaK', 'GammaS', 'lambdaK', 'lambdaS', 'deltaK', 'deltaS']:
            if param_name in params:
                print(f"  {param_name}: {params[param_name]}")

    filename = name.lower().replace(" ", "_") + ".txt"
    with open(os.path.join(results_dir, filename), 'w') as f:
        f.write(f"{name}\n")
        f.write("="*50 + "\n\n")
        
        f.write("Core Parameters:\n")
        for param_name in ['ak', 'bk', 'bs', 'k0', 'k1', 'n', 'p']:
            if param_name in params:
                f.write(f"  {param_name}: {params[param_name]}\n")
        
        if 'GammaK' in params:
            f.write("\nExtended Parameters (Suel et al.):\n")
            for param_name in ['GammaK', 'GammaS', 'lambdaK', 'lambdaS', 'deltaK', 'deltaS']:
                if param_name in params:
                    f.write(f"  {param_name}: {params[param_name]}\n")

def save_results(results, filename, results_dir):

    filepath = os.path.join(results_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {filepath}")

def load_results(filepath):

    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    
    return results

def is_excitable(params, model_odes, K_range=np.linspace(0, 1.5, 300), 
                S_range=np.linspace(0, 1.5, 300), verbose=False,
                find_fixed_points_func=None, classify_fixed_point_func=None):

    if find_fixed_points_func is None or classify_fixed_point_func is None:
        import competence_circuit_analysis as comp_model
        find_fixed_points_func = comp_model.find_fixed_points
        classify_fixed_point_func = comp_model.classify_fixed_point

    fps = find_fixed_points_func(params)
    
    if not fps:
        if verbose:
            print("No fixed points found.")
        return False, {}
    
    # Fixpunkt klassifizieren
    fp_types = [classify_fixed_point_func(fp[0], fp[1], params) for fp in fps]

    stable_fps = [fp for i, fp in enumerate(fps) if 'Stabil' in fp_types[i]]
    unstable_fps = [fp for i, fp in enumerate(fps) if 'Instabil' in fp_types[i]]
    saddle_fps = [fp for i, fp in enumerate(fps) if 'Sattel' in fp_types[i]]
    
    if verbose:
        print(f"Fixed points found: {len(fps)}")
        print(f"Stable fixed points: {len(stable_fps)}")
        print(f"Unstable fixed points: {len(unstable_fps)}")
        print(f"Saddle points: {len(saddle_fps)}")
        for i, (fp, fp_type) in enumerate(zip(fps, fp_types)):
            print(f"  FP{i+1}: ({fp[0]:.4f}, {fp[1]:.4f}) - {fp_type}")
    
    strict_excitable = (len(stable_fps) == 1 and 
                       len(saddle_fps) == 1 and 
                       len(unstable_fps) == 1 and
                       stable_fps[0][0] < 0.3)  # Stable point with low ComK
    
    loose_excitable = (len(stable_fps) == 1 and
                     len(saddle_fps) >= 1 and
                     len(fps) >= 3 and
                     stable_fps[0][0] < 0.3)  # Stable point with low ComK
    
   
    info = {
        'fixed_points': fps,
        'fp_types': fp_types,
        'stable_fps': stable_fps,
        'unstable_fps': unstable_fps,
        'saddle_fps': saddle_fps,
        'strict_excitable': strict_excitable,
        'loose_excitable': loose_excitable
    }
    
    return strict_excitable or loose_excitable, info

def identify_competence_events(t, K, threshold=0.5):

    competence_mask = K > threshold
    competence_events = []
    competence_durations = []
    rise_times = []
    peak_values = []
    
    if not competence_mask.any():
        return competence_events, competence_durations, rise_times, peak_values
    
    # Find transitions (0->1: start, 1->0: end)
    transitions = np.diff(competence_mask.astype(int))
    start_indices = np.where(transitions == 1)[0]
    end_indices = np.where(transitions == -1)[0]
    
    # Handle special cases
    if competence_mask[0]:
        start_indices = np.insert(start_indices, 0, 0)
    if competence_mask[-1]:
        end_indices = np.append(end_indices, len(competence_mask) - 1)
    
    # Ensure we have matching start and end points
    n = min(len(start_indices), len(end_indices))
    
    for i in range(n):
        start_idx = start_indices[i]
        end_idx = end_indices[i]
        start_t = t[start_idx]
        end_t = t[end_idx]
        
        # Only count events that are long enough (avoid noise artifacts)
        if end_t - start_t > 1.0: 
            event_K = K[start_idx:end_idx+1]
            event_t = t[start_idx:end_idx+1]
            
            peak_value = np.max(event_K)
            peak_idx = np.argmax(event_K)
            peak_t = event_t[peak_idx]
            
            rise_time = peak_t - start_t
         
            competence_events.append((start_t, end_t))
            competence_durations.append(end_t - start_t)
            rise_times.append(rise_time)
            peak_values.append(peak_value)
    
    return competence_events, competence_durations, rise_times, peak_values

def generate_ou_noise(theta, mu, sigma, dt):

    return np.random.normal(0, sigma * np.sqrt(dt))